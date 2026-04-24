"""
transcripts_semantic_chunked.py - Gộp chunks theo ngữ nghĩa
=================================================
Mục đích: Gộp các time chunks có liên quan về nội dung
           thành các semantic chunks lớn hơn

Pipeline: data/chunked/transcripts_time_chunked.jsonl -> data/chunked/transcripts_semantic_chunked.jsonl

Nguyên tắc gộp:
- Gộp khi 2 chunk có lexical overlap >= 8% (từ vựng chung)
- Gộp khi chunk tiếp theo bắt đầu bằng continuation: "and", "but", "so", etc.
- KHÔNG gộp khi có topic shift: "now let's", "next", "in summary", etc.
- Giới hạn: max 220 từ, max 120 giây

Tại sao cần semantic chunking:
- Time chunks chia theo thời gian cố định, không quan tâm nội dung
- Semantic chunks gộp các phần cùng chủ đề lại với nhau
- Khi retrieval, ta muốn lấy cả một "ý" chứ không phải 1/3 ý
"""

import json
import re
from pathlib import Path
from typing import Iterable

# =============================================================================
# CẤU HÌNH ĐƯỜNG DẪN
# =============================================================================
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_PATH = ROOT_DIR / "data" / "chunked" / "transcripts_time_chunked.jsonl"
OUTPUT_PATH = ROOT_DIR / "data" / "chunked" / "transcripts_semantic_chunked.jsonl"

# =============================================================================
# CẤU HÌNH GỘP CHUNK
# =============================================================================
# Số từ tối đa của semantic chunk (sau khi gộp)
MAX_SEMANTIC_WORDS = 220
# Thời gian tối đa của semantic chunk (giây)
MAX_SEMANTIC_DURATION = 120.0
# Số từ tối thiểu để xem là chunk "đủ lớn"
MIN_SEMANTIC_WORDS = 70
# Tỷ lệ từ vựng chung tối thiểu để gộp (8%)
MIN_LEXICAL_OVERLAP = 0.08

# =============================================================================
# STOPWORDS - Các từ phổ biến được bỏ qua khi tính overlap
# =============================================================================
STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "so",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
    "as",
    "at",
    "by",
    "from",
    "if",
    "then",
    "than",
    "into",
    "about",
    "also",
    "we",
    "you",
    "they",
    "he",
    "she",
    "i",
    "me",
    "our",
    "your",
    "their",
    "my",
    "them",
    "what",
    "which",
    "who",
    "when",
    "where",
    "why",
    "how",
    "do",
    "does",
    "did",
    "can",
    "could",
    "would",
    "should",
    "have",
    "has",
    "had",
    "not",
    "just",
    "there",
    "here",
}

# =============================================================================
# CONTINUATION PREFIXES - Các từ báo hiệu tiếp tục chủ đề
# =============================================================================
# Khi text bắt đầu bằng这些, ta gộp vào chunk trước (ý tiếp tục)
CONTINUATION_PREFIXES = (
    "and ",
    "but ",
    "so ",
    "because ",
    "then ",
    "now ",
    "for example",
    "for instance",
    "in other words",
    "that means",
    "this means",
    "which means",
    "however",
    "meanwhile",
)

# =============================================================================
# TOPIC SHIFT PREFIXES - Các từ báo hiệu chuyển chủ đề
# =============================================================================
# Khi text bắt đầu bằng这些, KHÔNG gộp (ý mới)
TOPIC_SHIFT_PREFIXES = (
    "now let's",
    "let's move",
    "moving on",
    "next ",
    "in summary",
    "to summarize",
    "the key idea",
)


def normalize_text(text: str) -> str:
    """
    Chuẩn hóa text (giống như time chunk).
    """
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;!?])", r"\1", text)
    text = re.sub(r"([,.;!?]){2,}", r"\1", text)
    text = text.strip(" -.,;:!?")
    if not text:
        return ""
    return text[0].upper() + text[1:]


def word_count(text: str) -> int:
    """Đếm số từ trong text."""
    return len(text.split())


def tokenize_content_words(text: str) -> set[str]:
    """
    Trích xuất các content words (bỏ stopwords).

    Args:
        text: Text đầu vào

    Returns:
        Set các từ có nghĩa (không phải stopwords, > 2 ký tự)
    """
    # Tìm tất cả các từ (chữ cái + số)
    tokens = re.findall(r"[a-zA-Z0-9']+", text.lower())
    # Lọc: bỏ stopwords và từ quá ngắn
    return {tok for tok in tokens if tok not in STOPWORDS and len(tok) > 2}


def lexical_overlap_score(text_a: str, text_b: str) -> float:
    """
    Tính lexical overlap giữa 2 texts.

    Công thức: |A ∩ B| / |A ∪ B| (Jaccard similarity)

    Ví dụ:
    - A = {"transformer", "attention", "model"}
    - B = {"transformer", "attention", "neural"}
    - overlap = 2 / 4 = 0.5 (50%)

    Args:
        text_a: Text thứ nhất
        text_b: Text thứ hai

    Returns:
        Tỷ lệ overlap (0.0 - 1.0)
    """
    a = tokenize_content_words(text_a)
    b = tokenize_content_words(text_b)
    if not a or not b:
        return 0.0
    # Jaccard: giao / hợp
    return len(a & b) / len(a | b)


def starts_with_continuation(text: str) -> bool:
    """
    Kiểm tra text có bắt đầu bằng continuation prefix.
    """
    lower = text.lower().strip()
    return any(lower.startswith(prefix) for prefix in CONTINUATION_PREFIXES)


def starts_with_topic_shift(text: str) -> bool:
    """
    Kiểm tra text có bắt đầu bằng topic shift prefix.
    """
    lower = text.lower().strip()
    return any(lower.startswith(prefix) for prefix in TOPIC_SHIFT_PREFIXES)


def should_merge(current: dict, nxt: dict) -> bool:
    """
    Quyết định có nên gộp 2 chunks hay không.

    Logic (OR - bất kỳ điều kiện nào đúng đều gộp):
    1. Chunk hiện tại < 70 từ -> gộp (chưa đủ lớn)
    2. Chunk tiếp theo < 35 từ -> gộp (quá ngắn)
    3. Text bắt đầu bằng continuation -> gộp (ý tiếp tục)
    4. Lexical overlap >= 8% -> gộp (cùng chủ đề)
    5. Text kết thúc bằng ":" -> gộp (danh sách)

    KHÔNG gộp khi:
    - Quá thời gian max
    - Quá số từ max
    - Có topic shift VÀ chunk trước đủ lớn

    Args:
        current: Chunk hiện tại
        nxt: Chunk tiếp theo

    Returns:
        True nên gộp, False nên tách
    """
    # Tính tổng duration và words nếu gộp
    merged_duration = float(nxt["end_time"]) - float(current["start_time"])
    merged_words = current["word_count"] + nxt["word_count"]

    # =============================================================================
    # RÀNG BUỘC: Không gộp nếu vượt giới hạn
    # =============================================================================
    if merged_duration > MAX_SEMANTIC_DURATION:
        return False
    if merged_words > MAX_SEMANTIC_WORDS:
        return False

    # Tính lexical overlap
    overlap = lexical_overlap_score(current["chunk_text"], nxt["chunk_text"])

    # =============================================================================
    # ĐIỀU KIỆN ĐẶC BIỆT: Topic shift không gộp
    # =============================================================================
    if (
        starts_with_topic_shift(nxt["chunk_text"])
        and current["word_count"] >= MIN_SEMANTIC_WORDS
    ):
        return False

    # =============================================================================
    # ĐIỀU KIỆN 1: Chunk trước quá ngắn -> gộp
    # =============================================================================
    if current["word_count"] < MIN_SEMANTIC_WORDS:
        return True

    # =============================================================================
    # ĐIỀU KIỆN 2: Chunk sau quá ngắn -> gộp
    # =============================================================================
    if nxt["word_count"] < 35:
        return True

    # =============================================================================
    # ĐIỀU KIỆN 3: Continuation -> gộp
    # =============================================================================
    if starts_with_continuation(nxt["chunk_text"]):
        return True

    # =============================================================================
    # ĐIỀU KIỆN 4: Overlap đủ lớn -> gộp
    # =============================================================================
    if overlap >= MIN_LEXICAL_OVERLAP:
        return True

    # =============================================================================
    # ĐIỀU KIỆN 5: Đang liệt kê -> gộp
    # =============================================================================
    if current["chunk_text"].rstrip().endswith(":"):
        return True

    return False


def merge_pair(current: dict, nxt: dict) -> dict:
    """
    Gộp 2 chunks thành 1.

    Args:
        current: Chunk hiện tại
        nxt: Chunk tiếp theo

    Returns:
        Chunk đã gộp
    """
    merged_text = normalize_text(current["chunk_text"] + " " + nxt["chunk_text"])
    return {
        "video_id": current["video_id"],
        "title": current.get("title", ""),
        "course": current.get("course", ""),
        "playlist_id": current.get("playlist_id", ""),
        "published_at": current.get("published_at", ""),
        "source": current.get("source", ""),
        "chunk_type": "semantic",
        "chunk_index": current["chunk_index"],
        # Lưu tham chiếu đến time chunks gốc
        "parent_time_chunk_start": current["parent_time_chunk_start"],
        "parent_time_chunk_end": nxt["parent_time_chunk_end"],
        "chunk_text": merged_text,
        "start_time": current["start_time"],
        "end_time": nxt["end_time"],
        "duration": round(float(nxt["end_time"]) - float(current["start_time"]), 3),
        "word_count": word_count(merged_text),
        "time_chunk_count": current["time_chunk_count"] + nxt["time_chunk_count"],
    }


def iter_jsonl(path: Path) -> Iterable[dict]:
    """Đọc file JSONL."""
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"SKIP line {line_no}: invalid JSON")


def group_by_video(path: Path) -> dict[str, list[dict]]:
    """
    Nhóm các chunks theo video_id.

    Args:
        path: Đường dẫn file time chunks

    Returns:
        Dict: {video_id: [chunks theo thứ tự]}
    """
    grouped = {}
    for row in iter_jsonl(path):
        # Thêm vào list của video tương ứng
        grouped.setdefault(row["video_id"], []).append(row)

    # Sắp xếp theo chunk_index
    for video_id in grouped:
        grouped[video_id].sort(key=lambda x: x["chunk_index"])
    return grouped


def build_semantic_chunks(time_chunks: list[dict]) -> list[dict]:
    """
    Gộp time chunks thành semantic chunks.

    Args:
        time_chunks: Danh sách time chunks (đã sắp xếp theo thứ tự)

    Returns:
        Danh sách semantic chunks
    """
    if not time_chunks:
        return []

    semantic_chunks = []
    # Khởi tạo chunk đầu tiên
    current = {
        **time_chunks[0],
        "chunk_type": "semantic",
        "parent_time_chunk_start": time_chunks[0]["chunk_index"],
        "parent_time_chunk_end": time_chunks[0]["chunk_index"],
        "time_chunk_count": 1,
    }

    # Duyệt qua các chunk tiếp theo
    for nxt_raw in time_chunks[1:]:
        nxt = {
            **nxt_raw,
            "chunk_type": "semantic",
            "parent_time_chunk_start": nxt_raw["chunk_index"],
            "parent_time_chunk_end": nxt_raw["chunk_index"],
            "time_chunk_count": 1,
        }

        # Kiểm tra có nên gộp không
        if should_merge(current, nxt):
            current = merge_pair(current, nxt)
        else:
            # Lưu chunk hiện tại và bắt đầu chunk mới
            current["chunk_index"] = len(semantic_chunks)
            semantic_chunks.append(current)
            current = nxt

    # Lưu chunk cuối cùng
    current["chunk_index"] = len(semantic_chunks)
    semantic_chunks.append(current)
    return semantic_chunks


def write_jsonl(records: Iterable[dict], path: Path) -> None:
    """Ghi records ra JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    print(f"Wrote {count} chunks -> {path}")


def main() -> None:
    """Hàm chính - xử lý tất cả videos."""
    # Nhóm chunks theo video
    grouped = group_by_video(INPUT_PATH)
    output_records = []

    # Gộp chunks trong mỗi video
    for _, time_chunks in grouped.items():
        output_records.extend(build_semantic_chunks(time_chunks))

    # Ghi output
    write_jsonl(output_records, OUTPUT_PATH)
    print(f"Processed videos: {len(grouped)}")
    print(f"Total semantic chunks: {len(output_records)}")


if __name__ == "__main__":
    main()
