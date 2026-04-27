"""
core/config.py – Cấu hình toàn cục cho RAG pipeline.
"""
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Settings:
    """Cấu hình toàn cục."""

    # ── Đường dẫn ───────────────────────────────────────────────────────────
    ROOT_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = Path(__file__).parent.parent / "embed/files"
    INDEX_DIR: Path = Path(__file__).parent.parent / "index"

    # ── Input file ───────────────────────────────────────────────────────────
    # File JSONL chứa semantic chunks đã qua xử lý từ Member 1
    raw_data_path: Path = DATA_DIR / "transcripts_enhanced.jsonl"

    # ── Output files (persist sau khi build) ─────────────────────────────────
    faiss_index_dir: Path = INDEX_DIR / "faiss"  # LangChain FAISS lưu toàn bộ vào folder
    bm25_path: Path = INDEX_DIR / "bm25.pkl"
    chunk_texts_path: Path = INDEX_DIR / "chunk_texts.pkl"

    # ── Embedding model ──────────────────────────────────────────────────────
    # BAAI/bge-m3: đa ngôn ngữ (VN↔EN), context 8192 tokens, output dim 1024
    embedding_model_name: str = "BAAI/bge-m3"
    embedding_dim: int = 1024

    # Device: "cpu" hoặc "cuda" (nếu có GPU)
    device: str = "cpu"  # thay thành "cuda" nếu có GPU

    # Normalize embeddings cho cosine similarity via inner product (dùng cho FAISS)
    normalize_embeddings: bool = True

    # ── Embedding batch size ─────────────────────────────────────────────────
    # Giảm xuống 16 nếu bị OOM (out of memory)
    embedding_batch_size: int = 32

    # ── Search parameters ────────────────────────────────────────────────────
    top_k_dense: int = 20  # lấy top-K từ FAISS trước khi rerank
    top_k_bm25: int = 20  # lấy top-K từ BM25
    top_k_final: int = 5  # trả về K kết quả cuối cùng

    # Hybrid alpha: alpha * dense_score + (1-alpha) * bm25_score
    hybrid_alpha: float = 0.6

    # ── LangChain setup ──────────────────────────────────────────────────────
    # LangChain FAISS dùng mặc định InnerProduct (IP) metric
    # (nếu normalize_embeddings=True thì IP ≈ cosine similarity)
    faiss_metric_type: str = "ip"  # "l2" hoặc "ip"


def get_settings() -> Settings:
    """
    Lấy cấu hình toàn cục.

    Có thể override từ environment variables:
        export EMBEDDING_MODEL=BAAI/bge-m3
        export DEVICE=cuda
        export BATCH_SIZE=64
    """
    import os

    cfg = Settings()

    # Override từ env nếu có
    if env_model := os.getenv("EMBEDDING_MODEL"):
        cfg.embedding_model_name = env_model

    if env_device := os.getenv("DEVICE"):
        cfg.device = env_device

    if env_batch := os.getenv("BATCH_SIZE"):
        cfg.embedding_batch_size = int(env_batch)

    if env_index := os.getenv("INDEX_DIR"):
        cfg.INDEX_DIR = Path(env_index)

    return cfg
