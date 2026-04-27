"""
core/embeddings.py – Khởi tạo embedding model.
"""
from langchain_huggingface import HuggingFaceEmbeddings
from .config import get_settings


def get_embedding_model():
    """
    Khởi tạo embedding model (bge-m3).

    bge-m3 được tối ưu cho:
      - Multilingual (VN, EN, ...)
      - Dense retrieval (semantic search)
      - Normalized embeddings (cosine similarity)

    Output:
        HuggingFaceEmbeddings object dùng được với LangChain
    """
    cfg = get_settings()

    embedder = HuggingFaceEmbeddings(
        model_name=cfg.embedding_model_name,
        model_kwargs={"device": cfg.device},
        encode_kwargs={"normalize_embeddings": cfg.normalize_embeddings},
    )

    return embedder
