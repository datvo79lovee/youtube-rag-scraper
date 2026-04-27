# Hướng dẫn sử dụng build_index.py

## ⚡ Tóm tắt những gì tôi sửa

### 1. **Tạo module `core/` mới**
   - `core/config.py` — Lưu cấu hình toàn cục (model, đường dẫn, ...)
   - `core/embeddings.py` — Khởi tạo embedding model bge-m3
   - `core/utils.py` — Các utility function (normalize_text, format_timestamp, ...)

### 2. **Tạo `data/build_index.py` chính thức**
   - Sửa import để dùng `core/` modules
   - Đảm bảo đúng LangChain API
   - Thêm comments chi tiết giải thích từng bước

### 3. **LangChain API fixes**
   - ✅ `FAISS.from_documents()` không có `batch_size` parameter
   - ✅ Dùng `add_documents()` để thêm batch tiếp theo
   - ✅ `save_local(folder)` tự động tạo `index.faiss` + `index.pkl`

---

## 🚀 Cách chạy

### Chạy với cấu hình mặc định
```bash
cd /Users/carwyn/youtube-rag-scraper
python -m data.build_index
```

### Chạy với file input/output riêng
```bash
python -m data.build_index \
  --data embed/files/transcripts_enhanced.jsonl \
  --out index
```

### Chạy với GPU (nếu có CUDA)
```bash
export DEVICE=cuda
python -m data.build_index
```

### Chạy với batch size nhỏ hơn (nếu bị OOM)
```bash
export BATCH_SIZE=16
python -m data.build_index
```

---

## 📁 Output structure

Sau khi chạy xong, bạn sẽ có:
```
index/
  ├─ faiss/
  │  ├─ index.faiss          ← Vector index (binary)
  │  └─ index.pkl            ← Docstore (metadata + document text)
  ├─ bm25.pkl                ← BM25 index
  └─ chunk_texts.pkl         ← Original texts (for debug)
```

---

## ⚠️ Vấn đề đã biết & cách fix

### 1. **ImportError: No module named 'core'**
   **Nguyên nhân:** Thư mục venv hoặc Python path chưa đúng
   **Fix:**
   ```bash
   source venv/bin/activate
   cd /Users/carwyn/youtube-rag-scraper
   python -m data.build_index
   ```

### 2. **ImportError: No module named 'langchain_community'**
   **Nguyên nhân:** Chưa cài đặt langchain-community
   **Fix:**
   ```bash
   pip install langchain-community
   ```

### 3. **OOM (Out of Memory) khi embed**
   **Nguyên nhân:** Batch size quá lớn
   **Fix:**
   ```bash
   export DEVICE=cpu
   export BATCH_SIZE=8
   python -m data.build_index
   ```

### 4. **FAISS index size lớn (>500MB)**
   **Nguyên nhân:** Bình thường cho 18k documents
   **Lưu ý:** Lần đầu chạy chậm, lần sau load nhanh

---

## 📊 Thời gian chạy dự kiến

- **CPU (i7 gen 10+):** 30-45 phút
- **GPU (T4/A100):** 3-8 phút
- **Embedding model:** BAAI/bge-m3 (1024 dims)
- **Input:** 18,395 documents (~30-50MB JSONL)

---

## 🔍 Debugging tips

### Xem log chi tiết
```bash
python -m data.build_index 2>&1 | tee build.log
```

### Kiểm tra data quality ngay
```python
from data.build_index import load_jsonl, analyze_data_quality
from pathlib import Path

records = load_jsonl(Path("embed/files/transcripts_enhanced.jsonl"))
analyze_data_quality(records)
```

### Kiểm tra FAISS index sau khi build
```python
from langchain_community.vectorstores import FAISS

vs = FAISS.load_local("index/faiss", embeddings_model)
print(f"Total vectors: {vs.index.ntotal}")
print(f"Vector dim: {vs.index.d}")
```

---

## 💡 Key points

1. **embedding_text vs chunk_text:**
   - `embedding_text` = header (course/video/time) + nội dung → dùng để EMBED
   - `chunk_text` = nội dung gốc → dùng để hiển thị + LLM prompt

2. **Tại sao cần hybrid (FAISS + BM25):**
   - FAISS tốt cho semantic similarity (ý nghĩa tương đồng)
   - BM25 tốt cho exact keyword matching
   - Kết hợp → recall cao + precision cao

3. **Normalize embeddings:**
   - `bge-m3` được set `normalize_embeddings=True`
   - → Inner product (IP) ≈ Cosine similarity
   - → Tốt cho cosine-based ranking

---

## ✅ Kiểm tra sau khi build thành công

```bash
# 1. Kiểm tra file output tồn tại
ls -lh index/

# 2. Kiểm tra size
du -sh index/

# 3. Load và test retrieval
python << 'EOF'
from langchain_community.vectorstores import FAISS
from core.embeddings import get_embedding_model

embedder = get_embedding_model()
vs = FAISS.load_local("index/faiss", embedder)
print(f"✅ FAISS loaded: {vs.index.ntotal} vectors")

# Test similarity search
results = vs.similarity_search("transformers training", k=3)
for r in results:
    print(f"  - {r.metadata['title']}: {r.page_content[:100]}...")
EOF
```

---

## Tiếp theo?

Sau khi build xong index, bạn có thể:
1. Tạo retriever (hybrid FAISS + BM25)
2. Build RAG chain với LLM
3. Tạo web UI để query
