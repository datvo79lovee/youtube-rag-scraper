"""
config.py – Cấu hình toàn cục cho Member 2 pipeline.
Sửa các hằng số ở đây thay vì scatter khắp nơi.
"""
from pathlib import Path

# ── Đường dẫn ───────────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).parent
DATA_DIR        = ROOT_DIR / "data"
INDEX_DIR       = ROOT_DIR / "index"
CHUNKS_FILE     = DATA_DIR / "semantic_chunks.jsonl"   # copy từ Member 1 vào đây

# Files được persist sau khi build
FAISS_INDEX_PATH    = INDEX_DIR / "faiss.index"
METADATA_PATH       = INDEX_DIR / "metadata.parquet"
BM25_PATH           = INDEX_DIR / "bm25.pkl"
EMBEDDINGS_NPY_PATH = INDEX_DIR / "embeddings.npy"

# ── Model ────────────────────────────────────────────────────────────────────
# BAAI/bge-m3: hỗ trợ cross-lingual VN↔EN, context 8192 tokens
EMBEDDING_MODEL = "BAAI/bge-m3"

# Fallback nếu RAM thấp (< 8GB)
# EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

EMBEDDING_DIM   = 1024          # bge-m3 output dim
BATCH_SIZE      = 32            # giảm xuống 16 nếu OOM
MAX_SEQ_LENGTH  = 512           # cắt bớt nếu chunk dài hơn

# ── Search ───────────────────────────────────────────────────────────────────
TOP_K_DENSE     = 20            # lấy top-K từ FAISS trước khi rerank
TOP_K_BM25      = 20            # lấy top-K từ BM25
TOP_K_FINAL     = 5             # trả về K kết quả cuối cùng

# Trọng số hybrid: alpha * dense_score + (1-alpha) * bm25_score
# alpha cao → ưu tiên semantic; alpha thấp → ưu tiên keyword
HYBRID_ALPHA    = 0.6

# ── Instruction prefix cho bge-m3 ─────────────────────────────────────────
# bge-m3 không cần instruction, nhưng nếu dùng bge-large-en-v1.5 thì cần
QUERY_INSTRUCTION = ""          # để trống với bge-m3

# ── YouTube URL template ──────────────────────────────────────────────────
YT_URL_TEMPLATE = "https://youtu.be/{video_id}?t={start_time}"
