"""
03_persist_index.py
Ngày 7: Persist BM25 index ra disk (FAISS + metadata đã được save ở script 01).
Script này chỉ cần chạy 1 lần sau khi script 01 đã xong.

Chạy:
    python scripts/03_persist_index.py
"""

import sys
import pickle
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from scripts.s02_hybrid_search import build_bm25, save_bm25


def main():
    # Kiểm tra artifacts từ script 01
    for p in [cfg.FAISS_INDEX_PATH, cfg.METADATA_PATH]:
        if not p.exists():
            sys.exit(
                f"[error] Missing: {p}\n"
                "  → Chạy 01_embed_and_index.py trước!"
            )

    print("[persist] Loading metadata...")
    meta = pd.read_parquet(cfg.METADATA_PATH)
    corpus_texts = meta["chunk_text"].tolist()

    # Build BM25 từ corpus
    print(f"[persist] Building BM25 trên {len(corpus_texts):,} docs...")
    bm25 = build_bm25(corpus_texts)
    save_bm25(bm25, cfg.BM25_PATH)

    # Verify FAISS
    index = faiss.read_index(str(cfg.FAISS_INDEX_PATH))
    print(f"\n✅ Tất cả artifacts đã được persist:")
    print(f"   FAISS index : {cfg.FAISS_INDEX_PATH}  ({index.ntotal:,} vectors)")
    print(f"   Metadata    : {cfg.METADATA_PATH}     ({len(meta):,} rows)")
    print(f"   BM25        : {cfg.BM25_PATH}")
    print(f"   Embeddings  : {cfg.EMBEDDINGS_NPY_PATH}")

    # In thống kê nhanh
    print("\n📊 Thống kê dataset:")
    print(meta.groupby("course")["video_id"].nunique().rename("videos").to_string())


if __name__ == "__main__":
    main()
