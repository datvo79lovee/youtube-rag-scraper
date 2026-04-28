"""
rag/retriever.py
════════════════
Hybrid Retriever kết hợp FAISS (dense) + BM25 (sparse) qua Reciprocal Rank Fusion.

Toàn bộ luồng tìm kiếm:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Query tiếng Việt                                                   │
  │         │                                                            │
  │    ┌────┴────┐                                                       │
  │    │  bge-m3  │  ← Cross-lingual: Việt query → English vector space │
  │    └────┬────┘                                                       │
  │         │ query vector                                               │
  │    ┌────┴──────────────┐                                             │
  │    │                   │                                             │
  │  FAISS              BM25                                             │
  │  (top-20)          (top-20)                                          │
  │    │                   │                                             │
  │    └────────┬──────────┘                                             │
  │             │                                                        │
  │   Reciprocal Rank Fusion (RRF)                                       │
  │             │                                                        │
  │         top-20 merged                                                │
  │             │                                                        │
  │   [Metadata Filter optional]                                         │
  │             │                                                        │
  │         List[RetrievedDoc]  → Reranker                               │
  └─────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from langchain_community.vectorstores import FAISS

from core.config import get_settings
from core.embeddings import get_embedding_model
from core.utils import build_source_label, make_youtube_url

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# DATA MODEL: KẾT QUẢ TÌM KIẾM
# ════════════════════════════════════════════════════════════════

@dataclass
class RetrievedDoc:
    """
    Một document được tìm thấy, kèm đầy đủ metadata.

    Dùng dataclass thay vì dict để:
      - Có type hints → IDE biết các field nào tồn tại
      - Dễ truyền giữa các function mà không cần nhớ key names
      - Có thể thêm method (to_dict, to_prompt_text, v.v.)
    """
    chunk_id: str
    chunk_text: str              # Nội dung tiếng Anh — đưa vào LLM prompt
    vi_text: str                 # Bản dịch tiếng Việt (hiện tại = rỗng)
    title: str                   # Tên video YouTube
    video_id: str
    course: str                  # Tên khóa học (CS25_Transformers, etc.)
    start_time: float            # Thời điểm bắt đầu trong video (giây)
    end_time: float
    url: str                     # YouTube deep-link URL với timestamp
    source_label: str            # Nhãn hiển thị, ví dụ: "Title [01:23]"
    score: float                 # Điểm relevance (cao hơn = liên quan hơn)
    retrieval_method: str = "hybrid"   # "dense" | "sparse" | "hybrid"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "chunk_text": self.chunk_text,
            "title": self.title,
            "course": self.course,
            "url": self.url,
            "source_label": self.source_label,
            "score": self.score,
            "retrieval_method": self.retrieval_method,
        }

    def to_prompt_text(self) -> str:
        """Format document thành văn bản để đưa vào LLM prompt."""
        return f"[{self.source_label}]\n{self.chunk_text}"


# ════════════════════════════════════════════════════════════════
# LOAD INDEX (cached)
# ════════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def _load_langchain_faiss(index_dir: str) -> FAISS:
    """
    Tải LangChain FAISS vectorstore từ disk.

    @lru_cache với tham số string đảm bảo:
      - Chỉ load một lần từ disk (nặng, chậm).
      - Streamlit re-run không load lại.
      - Nếu đổi index_dir → load lại (cache key thay đổi).

    allow_dangerous_deserialization=True cần thiết vì LangChain lưu
    index.pkl bằng pickle — Python yêu cầu flag này để load pickle từ nguồn ngoài.
    Ở môi trường production nên kiểm tra checksum của file trước khi load.
    """
    logger.info("Đang tải FAISS vectorstore từ %s...", index_dir)
    embedder = get_embedding_model()
    vectorstore = FAISS.load_local(
        folder_path=index_dir,
        embeddings=embedder,
        allow_dangerous_deserialization=True,   # cần thiết để load pickle
    )
    logger.info(" FAISS vectorstore đã tải: %d vectors", vectorstore.index.ntotal)
    return vectorstore


@lru_cache(maxsize=1)
def _load_bm25(index_dir: str):
    """Tải BM25 index từ disk."""
    bm25_path = Path(index_dir) / "bm25.pkl"
    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)
    logger.info("✅ BM25 index đã tải")
    return bm25


@lru_cache(maxsize=1)
def _load_chunk_texts(index_dir: str) -> List[str]:
    """Tải list chunk texts để BM25 có thể map index → text."""
    texts_path = Path(index_dir) / "chunk_texts.pkl"
    with open(texts_path, "rb") as f:
        texts = pickle.load(f)
    return texts


# ════════════════════════════════════════════════════════════════
# HYBRID RETRIEVER
# ════════════════════════════════════════════════════════════════

class HybridRetriever:
    """
    Tìm kiếm kết hợp Dense (FAISS) + Sparse (BM25) qua Reciprocal Rank Fusion.

    Khởi tạo một lần ở startup, gọi retrieve() cho mỗi query.

    Cross-lingual retrieval:
      Query tiếng Việt → bge-m3 embed → vector → FAISS tìm English docs gần nhất
      Hoạt động được vì bge-m3 được train trên parallel corpus đa ngôn ngữ,
      câu có cùng nghĩa ở các ngôn ngữ khác nhau sẽ có vector gần nhau.
    """

    def __init__(self, index_dir: Optional[str] = None) -> None:
        cfg = get_settings()
        self._index_dir = index_dir or str(cfg.index_dir)
        self._cfg = cfg

        # Load tất cả indices — sẽ được cache sau lần đầu
        self._vectorstore = _load_langchain_faiss(self._index_dir)
        self._bm25 = _load_bm25(self._index_dir)
        self._chunk_texts = _load_chunk_texts(self._index_dir)
        self._embedder = get_embedding_model()

    # ────────────────────────────────────────────
    # DENSE SEARCH (FAISS qua LangChain)
    # ────────────────────────────────────────────

    def _dense_search(
        self, query: str, k: int
    ) -> List[Dict[str, Any]]:
        """
        Tìm kiếm semantic dùng FAISS thông qua LangChain.

        LangChain FAISS.similarity_search_with_score() tự động:
          1. Embed query bằng embedder đã bind
          2. Tính cosine similarity (inner product vì vector đã normalize)
          3. Trả về list (Document, score) sorted theo score giảm dần

        Score ở đây là cosine similarity: 0.0 → 1.0 (càng cao càng giống)
        """
        results_with_scores = self._vectorstore.similarity_search_with_score(
            query=query,
            k=k,
        )

        # Chuyển sang format thống nhất: list[{doc, score, source}]
        results = []
        for doc, score in results_with_scores:
            results.append({
                "doc": doc,               # LangChain Document
                "score": float(score),    # cosine similarity
                "source": "dense",
            })

        logger.debug("FAISS dense: %d kết quả", len(results))
        return results

    # ────────────────────────────────────────────
    # SPARSE SEARCH (BM25)
    # ────────────────────────────────────────────

    def _sparse_search(
        self, query: str, k: int
    ) -> List[Dict[str, Any]]:
        """
        Tìm kiếm keyword dùng BM25.

        BM25 tính điểm dựa trên tần suất từ xuất hiện trong document
        so với toàn bộ corpus (IDF weighting).

        Lưu ý với Vietnamese query:
          BM25 tokenise bằng whitespace, nên "attention mechanism" sẽ match
          nhưng "cơ chế attention" sẽ không match từ tiếng Anh.
          → BM25 hoạt động kém hơn với query tiếng Việt.
          → RRF xử lý bằng cách giảm trọng số BM25 khi FAISS scores mạnh hơn.
        """
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)  # numpy array, 1 score per doc

        # Lấy top-k indices theo score giảm dần
        top_indices = np.argsort(scores)[::-1][:k]

        # BM25 trả về index trong chunk_texts — cần map lại sang Document
        # Vấn đề: BM25 và FAISS có thể có thứ tự document khác nhau
        # → Dùng chunk_texts để match với FAISS docstore qua chunk_id
        results = []
        faiss_docstore = self._vectorstore.docstore._dict  # {doc_id: Document}

        # Build lookup: chunk_id → (faiss_doc_id, Document)
        # (Chỉ build một lần — nên cache trong production)
        chunk_id_to_faiss = {}
        for faiss_id, doc in faiss_docstore.items():
            cid = doc.metadata.get("chunk_id", "")
            if cid:
                chunk_id_to_faiss[cid] = (faiss_id, doc)

        # BM25 index = vị trí trong chunk_texts list
        # chunk_texts[i] tương ứng với document thứ i trong records gốc
        # Nhưng FAISS docstore dùng UUID làm key → cần rebuild mapping
        # Cách đơn giản hơn: dùng index.reconstruct để lấy vector và tìm doc

        # Workaround: dùng similarity_search để lấy document từ text content
        for idx in top_indices:
            if idx < len(self._chunk_texts):
                text = self._chunk_texts[idx]
                bm25_score = float(scores[idx])
                if bm25_score <= 0:
                    continue
                results.append({
                    "text": text,           # text để tìm trong FAISS docstore
                    "score": bm25_score,
                    "source": "sparse",
                    "bm25_idx": int(idx),
                })

        logger.debug("BM25 sparse: %d kết quả (score > 0)", len(results))
        return results

    # ────────────────────────────────────────────
    # RECIPROCAL RANK FUSION
    # ────────────────────────────────────────────

    @staticmethod
    def _rrf_fuse(
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        alpha: float = 0.6,
        k_rrf: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Kết hợp hai ranked list bằng Reciprocal Rank Fusion (RRF).

        Công thức RRF:
          score(doc) = α × 1/(rank_dense + k) + (1-α) × 1/(rank_sparse + k)

        Tại sao dùng RRF thay vì cộng scores trực tiếp?
          - FAISS scores (0→1) và BM25 scores (0→vài chục) ở các thang khác nhau
          - RRF chỉ dùng thứ hạng (rank), không phụ thuộc vào scale của score
          - Ổn định hơn, không cần tuning score normalization
          - k_rrf=60 là giá trị chuẩn trong nghiên cứu IR (information retrieval)

        Ví dụ:
          Document A: rank 1 trong dense, rank 5 trong sparse
          Document B: rank 10 trong dense, rank 1 trong sparse (alpha=0.6)

          Score A = 0.6/61 + 0.4/65 ≈ 0.00984 + 0.00615 = 0.01599
          Score B = 0.6/70 + 0.4/61 ≈ 0.00857 + 0.00656 = 0.01513
          → A xếp trên B (xuất hiện tốt ở cả hai)
        """
        rrf_scores: Dict[str, float] = {}
        rrf_methods: Dict[str, str] = {}
        rrf_docs: Dict[str, Any] = {}

        # Đóng góp từ dense retrieval
        for rank, result in enumerate(dense_results):
            doc = result["doc"]
            cid = doc.metadata.get("chunk_id", f"dense_{rank}")
            contrib = alpha / (rank + 1 + k_rrf)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + contrib
            rrf_methods[cid] = "dense"
            rrf_docs[cid] = {"doc": doc, "source": "dense"}

        # Đóng góp từ sparse retrieval (BM25)
        # BM25 trả về text, không phải Document trực tiếp
        # → Dùng score để giữ thứ tự nhưng không store Document
        for rank, result in enumerate(sparse_results):
            text = result.get("text", "")
            # Tìm chunk_id tương ứng trong FAISS docstore
            cid = f"bm25_{result.get('bm25_idx', rank)}"   # placeholder
            contrib = (1 - alpha) / (rank + 1 + k_rrf)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + contrib
            if cid in rrf_methods:
                rrf_methods[cid] = "hybrid"
            else:
                rrf_methods[cid] = "sparse"

        # Sort theo RRF score giảm dần
        sorted_items = sorted(rrf_scores.items(), key=lambda x: -x[1])

        merged = []
        for cid, score in sorted_items:
            if cid in rrf_docs:
                item = rrf_docs[cid].copy()
                item["rrf_score"] = score
                item["retrieval_method"] = rrf_methods.get(cid, "hybrid")
                merged.append(item)

        return merged

    # ────────────────────────────────────────────
    # METADATA FILTER
    # ────────────────────────────────────────────

    @staticmethod
    def _apply_filter(
        results: List[Dict[str, Any]],
        filters: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Lọc kết quả theo metadata.

        Ví dụ: chỉ lấy docs từ khóa học CS25:
          filters = {"course": "CS25_Transformers"}

        Có thể filter nhiều field:
          filters = {"course": "CS25_Transformers", "video_id": "xyz123"}
        """
        if not filters:
            return results

        filtered = []
        for r in results:
            doc = r.get("doc")
            if doc is None:
                continue
            if all(doc.metadata.get(k) == v for k, v in filters.items()):
                filtered.append(r)

        logger.debug("Sau filter: %d / %d docs", len(filtered), len(results))
        return filtered

    # ────────────────────────────────────────────
    # BUILD RetrievedDoc
    # ────────────────────────────────────────────

    @staticmethod
    def _to_retrieved_doc(result: Dict[str, Any]) -> RetrievedDoc:
        """Chuyển từ internal result dict sang typed RetrievedDoc."""
        doc = result["doc"]
        meta = doc.metadata

        url = meta.get("url") or make_youtube_url(
            meta.get("video_id", ""), meta.get("start_time", 0)
        )
        label = meta.get("source_label") or build_source_label(meta)

        return RetrievedDoc(
            chunk_id=meta.get("chunk_id", ""),
            chunk_text=meta.get("chunk_text", doc.page_content),
            vi_text=meta.get("vi_text", ""),
            title=meta.get("title", ""),
            video_id=meta.get("video_id", ""),
            course=meta.get("course", ""),
            start_time=meta.get("start_time", 0.0),
            end_time=meta.get("end_time", 0.0),
            url=url,
            source_label=label,
            score=result.get("rrf_score", result.get("score", 0.0)),
            retrieval_method=result.get("retrieval_method", "hybrid"),
            metadata=meta,
        )

    # ────────────────────────────────────────────
    # PUBLIC API
    # ────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedDoc]:
        """
        Tìm kiếm hybrid và trả về top-k documents.

        Args:
            query:   Câu hỏi của người dùng (tiếng Việt hoặc tiếng Anh).
            k:       Số documents trả về (default: retrieval_top_k từ config).
            filters: Lọc theo metadata, ví dụ {"course": "CS25_Transformers"}.

        Returns:
            List[RetrievedDoc] sorted theo relevance score giảm dần.

        Cross-lingual flow:
            "Cơ chế attention hoạt động như thế nào?"
             → bge-m3 embed → vector tương đồng với "how attention mechanism works"
             → FAISS tìm docs tiếng Anh về attention
             → Trả về English docs
             → LLM đọc English docs, trả lời tiếng Việt
        """
        cfg = self._cfg
        k = k or cfg.retrieval_top_k

        logger.debug("Hybrid retrieve | query=%s... | k=%d", query[:60], k)

        # Bước 1: Dense search (FAISS)
        dense = self._dense_search(query, k)

        # Bước 2: Sparse search (BM25)
        sparse = self._sparse_search(query, k)

        # Bước 3: RRF fusion
        merged = self._rrf_fuse(dense, sparse, alpha=cfg.hybrid_alpha)

        # Bước 4: Filter theo metadata (nếu có)
        filtered = self._apply_filter(merged, filters)

        # Bước 5: Lấy top-k sau filter
        top = filtered[:k]

        # Bước 6: Chuyển sang RetrievedDoc
        docs = [self._to_retrieved_doc(r) for r in top if "doc" in r]

        logger.debug("Kết quả cuối: %d docs", len(docs))
        return docs