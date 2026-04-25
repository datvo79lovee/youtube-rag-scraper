"""
02_hybrid_search.py
Ngày 6: Hybrid Search = BM25 (sparse) + FAISS dense vector, kết hợp bằng RRF hoặc linear fusion.

Tại sao cần Hybrid?
- Dense (bge-m3): hiểu ngữ nghĩa, cross-lingual VN→EN, tốt với câu hỏi dài
- BM25: chính xác với technical terms (backpropagation, BLEU score, transformer...)
  mà embedding hay "làm mờ" đi

Chạy interactive:
    python scripts/02_hybrid_search.py --query "attention mechanism hoạt động như thế nào"
    python scripts/02_hybrid_search.py --query "gradient descent" --alpha 0.4 --top-k 5
"""

import argparse
import pickle
import re
import sys
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


# ─── Tokenizer đơn giản ──────────────────────────────────────────────────────
def tokenize(text: str) -> list[str]:
    """
    Lowercase + split. Giữ dấu gạch nối (self-attention, fine-tuning...).
    Đủ tốt cho BM25 trên transcript tiếng Anh.
    """
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text)
    return tokens


# ─── BM25 Builder ─────────────────────────────────────────────────────────────
def build_bm25(corpus_texts: list[str]) -> BM25Okapi:
    tokenized = [tokenize(t) for t in corpus_texts]
    return BM25Okapi(tokenized)


def save_bm25(bm25: BM25Okapi, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bm25, f)
    print(f"[bm25] Saved → {path}")


def load_bm25(path: Path) -> BM25Okapi:
    with open(path, "rb") as f:
        return pickle.load(f)


# ─── Score normalization ──────────────────────────────────────────────────────
def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """Đưa scores về [0, 1]. Safe với array toàn bằng nhau."""
    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-10:
        return np.zeros_like(scores)
    return (scores - mn) / (mx - mn)


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    k: int = 60,
) -> dict[int, float]:
    """
    RRF score = sum_over_lists( 1 / (k + rank) )
    Robust hơn linear fusion khi score scales khác nhau.
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


# ─── HybridSearcher ───────────────────────────────────────────────────────────
class HybridSearcher:
    """
    Kết hợp dense (FAISS) + sparse (BM25) retrieval.

    Hai chế độ fusion:
      - 'linear': alpha * dense_norm + (1-alpha) * bm25_norm
      - 'rrf'   : Reciprocal Rank Fusion (không cần tune alpha)
    """

    def __init__(
        self,
        faiss_index: faiss.Index,
        bm25: BM25Okapi,
        metadata_df: pd.DataFrame,
        model: SentenceTransformer,
        alpha: float = cfg.HYBRID_ALPHA,
        fusion: str = "linear",   # "linear" hoặc "rrf"
    ):
        self.index    = faiss_index
        self.bm25     = bm25
        self.meta     = metadata_df
        self.model    = model
        self.alpha    = alpha
        self.fusion   = fusion
        self.n_docs   = len(metadata_df)

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed query thành vector float32 shape (1, dim)."""
        prefix = cfg.QUERY_INSTRUCTION + query if cfg.QUERY_INSTRUCTION else query
        vec = self.model.encode(
            [prefix],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec.astype(np.float32)

    def _dense_search(self, query_vec: np.ndarray, top_k: int):
        """FAISS inner product search → (scores, indices)."""
        scores, indices = self.index.search(query_vec, top_k)
        return scores[0], indices[0]   # squeeze batch dim

    def _bm25_search(self, query: str, top_k: int):
        """BM25 search → sorted (score, idx) pairs."""
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)          # shape (N,)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return scores[top_indices], top_indices

    def search(
        self,
        query: str,
        top_k: int = cfg.TOP_K_FINAL,
        top_k_dense: int = cfg.TOP_K_DENSE,
        top_k_bm25: int = cfg.TOP_K_BM25,
    ) -> pd.DataFrame:
        """
        Chạy hybrid search, trả về DataFrame top-k kết quả
        với các cột: rank, score, title, course, url, snippet.
        """
        query_vec = self._embed_query(query)

        # 1. Dense retrieval
        dense_scores, dense_ids = self._dense_search(query_vec, top_k_dense)

        # 2. Sparse BM25 retrieval
        bm25_scores, bm25_ids = self._bm25_search(query, top_k_bm25)

        if self.fusion == "rrf":
            # ── RRF fusion ──
            rrf = reciprocal_rank_fusion([
                list(dense_ids),
                list(bm25_ids),
            ])
            # Sắp xếp theo RRF score giảm dần
            sorted_ids = sorted(rrf, key=rrf.get, reverse=True)[:top_k]
            final_scores = [rrf[i] for i in sorted_ids]
        else:
            # ── Linear fusion ──
            # Gom tất cả doc_ids xuất hiện trong cả 2 list
            candidate_ids = list(set(dense_ids.tolist()) | set(bm25_ids.tolist()))

            # Map id → normalized score
            dense_norm_map = {
                idx: s for idx, s in zip(
                    dense_ids, min_max_normalize(dense_scores)
                )
            }
            bm25_norm_map = {
                idx: s for idx, s in zip(
                    bm25_ids, min_max_normalize(bm25_scores)
                )
            }

            combined = {}
            for idx in candidate_ids:
                d = dense_norm_map.get(idx, 0.0)
                b = bm25_norm_map.get(idx, 0.0)
                combined[idx] = self.alpha * d + (1 - self.alpha) * b

            sorted_ids = sorted(combined, key=combined.get, reverse=True)[:top_k]
            final_scores = [combined[i] for i in sorted_ids]

        # 3. Lấy metadata
        rows = []
        for rank, (doc_id, score) in enumerate(zip(sorted_ids, final_scores), start=1):
            if doc_id < 0 or doc_id >= self.n_docs:
                continue
            row = self.meta.iloc[doc_id]
            snippet = row["chunk_text"][:300].replace("\n", " ") + "..."
            rows.append({
                "rank":    rank,
                "score":   round(float(score), 4),
                "title":   row["title"],
                "course":  row["course"],
                "start":   f"{int(row['start_time'] // 60)}:{int(row['start_time'] % 60):02d}",
                "url":     row["url"],
                "snippet": snippet,
            })

        return pd.DataFrame(rows)


# ─── Factory: load từ disk ────────────────────────────────────────────────────
def load_searcher(
    alpha: float = cfg.HYBRID_ALPHA,
    fusion: str = "linear",
) -> HybridSearcher:
    """Load tất cả artifacts từ disk và trả về HybridSearcher."""
    for p in [cfg.FAISS_INDEX_PATH, cfg.METADATA_PATH, cfg.BM25_PATH]:
        if not p.exists():
            sys.exit(
                f"[error] Missing: {p}\n"
                f"  → Chạy 01_embed_and_index.py trước, sau đó 03_persist_index.py"
            )

    print("[load] Loading artifacts...")
    index = faiss.read_index(str(cfg.FAISS_INDEX_PATH))
    meta  = pd.read_parquet(cfg.METADATA_PATH)
    bm25  = load_bm25(cfg.BM25_PATH)

    print("[load] Loading embedding model...")
    model = SentenceTransformer(cfg.EMBEDDING_MODEL)
    model.max_seq_length = cfg.MAX_SEQ_LENGTH

    print(f"[load] Ready: {index.ntotal:,} vectors, {len(meta):,} docs")
    return HybridSearcher(index, bm25, meta, model, alpha=alpha, fusion=fusion)


# ─── CLI ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Hybrid search CLI")
    parser.add_argument("--query",   type=str, required=True)
    parser.add_argument("--top-k",  type=int, default=cfg.TOP_K_FINAL)
    parser.add_argument("--alpha",  type=float, default=cfg.HYBRID_ALPHA,
                        help="0=pure BM25, 1=pure dense")
    parser.add_argument("--fusion", type=str, default="linear",
                        choices=["linear", "rrf"])
    args = parser.parse_args()

    searcher = load_searcher(alpha=args.alpha, fusion=args.fusion)
    results  = searcher.search(args.query, top_k=args.top_k)

    print(f"\n🔍 Query: '{args.query}'\n")
    for _, row in results.iterrows():
        print(f"  [{row['rank']}] score={row['score']}  {row['course']} | {row['title']} @ {row['start']}")
        print(f"       {row['url']}")
        print(f"       {row['snippet'][:150]}...\n")


if __name__ == "__main__":
    main()
