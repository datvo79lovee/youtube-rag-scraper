# TỔNG HỢP NHỮNG SỬA ĐỔI VÀ HƯỚNG DẪN CHẠY

## 📋 Những gì tôi đã sửa

### ✅ Đã tạo module `core/`

| File | Mục đích |
|------|---------|
| `core/__init__.py` | Package marker |
| `core/config.py` | Cấu hình toàn cục (model, path, hyperparams) |
| `core/embeddings.py` | Khởi tạo embedding model HuggingFace |
| `core/utils.py` | Utility functions (normalize, format, ...) |

### ✅ Đã tạo `data/build_index.py`

- **Sửa import:** `from langchain.schema import Document` → `from langchain_core.documents import Document`
  - ✅ LangChain v1.x dùng `langchain_core`
  
- **Sửa FAISS API:**
  - ✅ `FAISS.from_documents()` KHÔNG có parameter `batch_size`
  - ✅ Dùng `add_documents()` để thêm batch tiếp theo
  - ✅ `save_local()` tự động tạo `index.faiss` + `index.pkl`

- **Thêm comments chi tiết:**
  - Giải thích từng bước
  - Lý do dùng `embedding_text` thay vì `chunk_text` để embed
  - Tại sao cần hybrid (FAISS + BM25)

### ✅ Đã cài dependencies

```bash
pip install langchain langchain-core langchain-community langchain-huggingface
```

---

## 🚀 CÁCH CHẠY BUILD INDEX

### **Cách 1: Chạy với cấu hình mặc định**

```bash
cd /Users/carwyn/youtube-rag-scraper
source venv/bin/activate

python -m data.build_index
```

**Output:** `index/faiss/` (index FAISS) + `index/bm25.pkl` (index BM25)

### **Cách 2: Chạy với custom paths**

```bash
python -m data.build_index \
  --data embed/files/transcripts_enhanced.jsonl \
  --out index
```

### **Cách 3: Chạy với GPU (nếu có)**

```bash
export DEVICE=cuda
python -m data.build_index
```

### **Cách 4: Chạy với batch size nhỏ (nếu bị OOM)**

```bash
export BATCH_SIZE=16
python -m data.build_index
```

### **Cách 5: Chạy và lưu log**

```bash
python -m data.build_index 2>&1 | tee build.log
```

---

## 📁 File structure sau khi build thành công

```
index/
├── faiss/
│   ├── index.faiss        ← Vector index (binary, ~200-300MB)
│   └── index.pkl          ← Docstore (metadata + text, ~150-200MB)
├── bm25.pkl               ← BM25 index (~20-30MB)
└── chunk_texts.pkl        ← Original texts backup (~50-100MB)
```

---

## ✅ Kiểm tra sau khi build

```bash
python3 << 'EOF'
from pathlib import Path
from langchain_community.vectorstores import FAISS
from core.embeddings import get_embedding_model

embedder = get_embedding_model()
index_path = Path("index/faiss")

# Load FAISS index
vs = FAISS.load_local(str(index_path), embedder)
print(f"✅ FAISS loaded successfully")
print(f"   - Total vectors: {vs.index.ntotal}")
print(f"   - Vector dimension: {vs.index.d}")

# Test similarity search
query = "transformers neural networks training"
results = vs.similarity_search(query, k=3)
print(f"\n🔍 Search results for '{query}':")
for i, doc in enumerate(results, 1):
    print(f"\n{i}. {doc.metadata['title']}")
    print(f"   Course: {doc.metadata['course']}")
    print(f"   Time: {doc.metadata['source_label'].split('[')[1].rstrip(']')}")
    print(f"   Preview: {doc.page_content[:150]}...")
EOF
```

---

## ⚠️ Các lỗi thường gặp & cách fix

### 1. `ModuleNotFoundError: No module named 'core'`
**Fix:** Chắc chắn bạn chạy từ thư mục `/Users/carwyn/youtube-rag-scraper`
```bash
cd /Users/carwyn/youtube-rag-scraper
source venv/bin/activate
python -m data.build_index
```

### 2. `ModuleNotFoundError: No module named 'langchain'`
**Fix:** Cài đặt LangChain packages
```bash
source venv/bin/activate
pip install langchain langchain-core langchain-community langchain-huggingface
```

### 3. `ModuleNotFoundError: No module named 'sentence_transformers'`
**Fix:** Đã có trong requirements.txt, nhưng cài lại chắc chắn:
```bash
pip install sentence-transformers
```

### 4. Lỗi CUDA / GPU memory
**Fix:** Dùng CPU + batch size nhỏ
```bash
export DEVICE=cpu
export BATCH_SIZE=8
python -m data.build_index
```

### 5. File transcripts_enhanced.jsonl không tìm thấy
**Fix:** Kiểm tra đường dẫn đúng
```bash
ls -lh embed/files/transcripts_enhanced.jsonl
```

---

## 💡 Hiểu về flow xử lý

### Step 1: Load JSONL
```
transcripts_enhanced.jsonl (18k records)
  ↓
  - chunk_id: unique ID
  - video_id: YouTube ID
  - chunk_text: nội dung gốc (English)
  - embedding_text: nội dung + header (English)
  - metadata: title, course, time, ...
```

### Step 2: Data quality check
```
Thống kê:
  - 18,395 records OK
  - Tất cả có chunk_text → OK để embed
  - Tất cả có embedding_text → dùng cái này để embed
  - 0% có chunk_vi → không có bản dịch Việt
  - Tất cả < 512 tokens → safe
  - Tất cả chunk_id unique → safe
```

### Step 3: Convert to LangChain Documents
```
Record JSONL → Document(page_content, metadata)
  
page_content = embedding_text  (dùng để embed)
metadata = {
  chunk_id, video_id, title, course, start_time, end_time,
  url, source_label, chunk_text (gốc), vi_text (nếu có)
}
```

### Step 4: Embed + Build FAISS
```
18,395 docs
  ↓ (batch 32)
  embed with bge-m3 (1024-dim vectors)
  ↓
  FAISS index (inner product metric)
  ↓
  save → index/faiss/
```

### Step 5: Build BM25
```
18,395 texts (chunk_text)
  ↓ (tokenize: lowercase + split)
  BM25Okapi index
  ↓
  save → index/bm25.pkl
```

---

## 🔗 Tiếp theo?

Sau khi build xong index, bạn có thể:

1. **Tạo hybrid retriever** (FAISS + BM25)
2. **Tạo RAG chain** (retriever → LLM → answer)
3. **Build web UI** (Streamlit/Gradio)
4. **Deploy** (Docker/Cloud)

Các file cần tạo tiếp theo:
- `core/retriever.py` — Hybrid retriever
- `core/rag_chain.py` — RAG chain
- `app/app.py` — Streamlit UI
- `docker-compose.yml` — Deploy setup

---

## 📊 Dự kiến thời gian & resource

| Hardware | CPU Time | GPU Time | RAM |
|----------|----------|----------|-----|
| i7 Gen 10+ | 30-45m | - | 16GB |
| M1 Mac | 25-35m | - | 16GB |
| T4 GPU | - | 3-8m | 8GB |
| A100 GPU | - | 1-3m | 8GB |

---

## ✨ Summary

✅ **Code đúng cú pháp LangChain** (v1.x)  
✅ **Import path đúng** (`langchain_core`, không `langchain.schema`)  
✅ **FAISS API đúng** (batch handling, save/load)  
✅ **Module structure sạch** (core/ module tách biệt)  
✅ **Sẵn sàng chạy** (test OK)  

🚀 **Ready to go!**
