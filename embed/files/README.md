# Member 2 – NLP & Embedding Pipeline

Hệ thống tìm kiếm hybrid (BM25 + Dense Vector) cho Stanford AI/ML lecture transcripts.

## Cấu trúc thư mục

```
member2/
├── README.md
├── requirements.txt
├── config.py                    # Cấu hình toàn cục
├── scripts/
│   ├── 01_embed_and_index.py    # Ngày 3-5: Build FAISS index từ chunks
│   ├── 02_hybrid_search.py      # Ngày 6: Hybrid BM25 + Vector search
│   ├── 03_persist_index.py      # Ngày 7: Persist & load index
│   └── 04_eval_retrieval.py     # Ngày 7: Eval 20 queries tiếng Việt
├── index/                       # Thư mục lưu FAISS index + metadata
├── data/
│   └── sample_queries/
│       └── vn_queries_20.json   # 20 câu hỏi tiếng Việt mẫu
└── tests/
    └── test_retrieval.py        # Unit tests
```

## Yêu cầu

```bash
pip install -r requirements.txt
```

## Chạy pipeline theo thứ tự

```bash
# Bước 1: Build FAISS index (cần GPU để nhanh hơn, CPU vẫn OK)
python scripts/01_embed_and_index.py --input data/semantic_chunks.jsonl

# Bước 2: Test hybrid search interactively
python scripts/02_hybrid_search.py --query "attention mechanism là gì"

# Bước 3: Persist index ra disk
python scripts/03_persist_index.py

# Bước 4: Chạy eval 20 queries tiếng Việt
python scripts/04_eval_retrieval.py
```

## Model được dùng

- **Embedding**: `BAAI/bge-m3` — hỗ trợ cross-lingual Vietnamese ↔ English tốt nhất
- **BM25**: `rank_bm25` — sparse retrieval cho technical terms (gradient descent, backprop...)
- **FAISS**: `IndexFlatIP` (inner product / cosine) — exact search, đủ nhanh với 14K chunks

## Notes

- Chunks đã được clean + semantic-chunked bởi Member 1, **không cần** xử lý thêm
- Mỗi chunk giữ metadata: `video_id`, `title`, `course`, `start_time`, `end_time`
- URL YouTube được reconstruct tự động: `https://youtu.be/{video_id}?t={start_time}`
