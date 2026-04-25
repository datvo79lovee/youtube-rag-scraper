"""
01_embed_and_index.py
Ngày 3–5: Load semantic chunks → embed bằng bge-m3 → build FAISS IndexFlatIP.

Chạy:
    python scripts/01_embed_and_index.py
    python scripts/01_embed_and_index.py --input /path/to/semantic_chunks.jsonl
    python scripts/01_embed_and_index.py --batch-size 16   # nếu RAM thấp
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Thêm root vào path để import config
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


# ─────────────────────────────────────────────────────────────────────────────
def load_chunks(jsonl_path: Path) -> list[dict]:
    """Load tất cả chunks từ JSONL, trả về list of dict."""
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    print(f"[load] Loaded {len(chunks):,} chunks từ {jsonl_path.name}")
    return chunks


def build_metadata_df(chunks: list[dict]) -> pd.DataFrame:
    """
    Tạo DataFrame metadata – chỉ giữ các cột cần thiết cho retrieval.
    URL YouTube được reconstruct tự động.
    """
    rows = []
    for i, c in enumerate(chunks):
        start = int(c.get("start_time", 0))
        rows.append({
            "chunk_idx":  i,
            "video_id":   c["video_id"],
            "title":      c["title"],
            "course":     c["course"],
            "playlist_id": c.get("playlist_id", ""),
            "published_at": c.get("published_at", ""),
            "start_time": c.get("start_time", 0.0),
            "end_time":   c.get("end_time", 0.0),
            "duration":   c.get("duration", 0.0),
            "word_count": c.get("word_count", 0),
            "chunk_index": c.get("chunk_index", 0),
            "chunk_text": c["chunk_text"],
            "url":        cfg.YT_URL_TEMPLATE.format(
                            video_id=c["video_id"], start_time=start),
        })
    df = pd.DataFrame(rows)
    print(f"[metadata] DataFrame shape: {df.shape}")
    return df


def embed_chunks(
    texts: list[str],
    model_name: str,
    batch_size: int,
    max_seq_length: int,
) -> np.ndarray:
    """
    Embed danh sách texts bằng SentenceTransformer.
    Trả về float32 numpy array shape (N, dim).
    """
    print(f"[embed] Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    model.max_seq_length = max_seq_length

    print(f"[embed] Encoding {len(texts):,} chunks (batch={batch_size})...")
    t0 = time.time()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # cần cho cosine via inner product
        convert_to_numpy=True,
    )
    elapsed = time.time() - t0
    print(f"[embed] Done in {elapsed:.1f}s  shape={embeddings.shape}  dtype={embeddings.dtype}")
    return embeddings.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build FAISS IndexFlatIP (inner product = cosine khi vectors đã normalize).
    Exact search – đủ nhanh với 14K vectors.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"[faiss] Index built: {index.ntotal:,} vectors, dim={dim}")
    return index


def save_artifacts(
    index: faiss.Index,
    metadata_df: pd.DataFrame,
    embeddings: np.ndarray,
) -> None:
    """Persist FAISS index, metadata parquet, và embeddings npy."""
    cfg.INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # FAISS index
    faiss.write_index(index, str(cfg.FAISS_INDEX_PATH))
    print(f"[save] FAISS index → {cfg.FAISS_INDEX_PATH}")

    # Metadata (Parquet – nhanh hơn CSV, preserve dtypes)
    metadata_df.to_parquet(cfg.METADATA_PATH, index=False)
    print(f"[save] Metadata    → {cfg.METADATA_PATH}")

    # Raw embeddings (giữ lại để debug / fine-tune sau)
    np.save(cfg.EMBEDDINGS_NPY_PATH, embeddings)
    print(f"[save] Embeddings  → {cfg.EMBEDDINGS_NPY_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Embed chunks & build FAISS index")
    parser.add_argument("--input", type=Path, default=cfg.CHUNKS_FILE,
                        help="Path to semantic_chunks.jsonl")
    parser.add_argument("--model", type=str, default=cfg.EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--max-seq-length", type=int, default=cfg.MAX_SEQ_LENGTH)
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"[error] File không tồn tại: {args.input}\n"
                 f"  → Copy semantic_chunks.jsonl vào {cfg.DATA_DIR}/")

    # Pipeline
    chunks      = load_chunks(args.input)
    metadata_df = build_metadata_df(chunks)
    texts       = [c["chunk_text"] for c in chunks]
    embeddings  = embed_chunks(texts, args.model, args.batch_size, args.max_seq_length)
    index       = build_faiss_index(embeddings)
    save_artifacts(index, metadata_df, embeddings)

    print("\n✅ Build hoàn tất!")
    print(f"   {index.ntotal:,} vectors | dim={embeddings.shape[1]}")
    print(f"   Index dir: {cfg.INDEX_DIR}")


if __name__ == "__main__":
    main()
