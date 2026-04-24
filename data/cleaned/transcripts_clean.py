"""
transcripts_clean.py - Làm sạch transcript YouTube
===============================================
Mục đích: Xóa noise, filler words, timestamp từ transcript thô

Pipeline: data/raw/transcripts.jsonl -> data/cleaned/transcripts_clean.jsonl

Các bước xử lý:
1. Xóa timestamp SRT/VTT (00:00:00 --> 00:00:00)
2. Xóa marker noise: [applause], [laughter], [music], [inaudible]
3. Xóa HTML tags: <font>, <c>, etc.
4. Xóa filler words: uh, um, you know, I mean
5. Chuẩn hóa whitespace + dấu câu
"""

import json
import re
from pathlib import Path

# =============================================================================
# CẤU HÌNH ĐƯỜNG DẪN
# =============================================================================
# ROOT_DIR xác định thư mục gốc của project tự động
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
# File input: transcript thô từ YouTube
INPUT_PATH = ROOT_DIR / "data" / "raw" / "transcripts.jsonl"
# File output: transcript đã làm sạch
OUTPUT_PATH = ROOT_DIR / "data" / "cleaned" / "transcripts_clean.jsonl"

# =============================================================================
# REGEX PATTERNS - Định nghĩa các pattern để tìm và thay thế
# =============================================================================

# Pattern 1: Xóa marker noise trong ngoặc vuông [applause], [laughter], etc.
# r"\[(?:inaudible|applause|laughter|music|silence|noise|crosstalk).*?\]"
# - \[\]: match ngoặc vuông
# - (?:...): non-capturing group
# - .*?: match any character (non-greedy)
# - re.IGNORECASE: không phân biệt hoa/thường
BRACKET_NOISE_RE = re.compile(
    r"\[(?:inaudible|applause|laughter|music|silence|noise|crosstalk).*?\]",
    re.IGNORECASE,
)

# Pattern 2: Xóa HTML tags <anything>
HTML_TAG_RE = re.compile(r"<[^>]+>")

# Pattern 3: Xóa timestamp dạng SRT/VTT "00:00:00 --> 00:00:00"
# \b: word boundary
# \d{1,2}: 1-2 digits (phút/giây)
# (?::\d{2})?: optional seconds với dấu :
# \s*-->*\s*: match --> với khoảng trắng
TIMESTAMP_RANGE_RE = re.compile(
    r"\b\d{1,2}:\d{2}(?::\d{2})?\s*-->\s*\d{1,2}:\d{2}(?::\d{2})?\b"
)

# Pattern 4: Xóa standalone timestamp "00:00" không có -->
# (?<!\d): negative lookbehind - không có digit đứng trước
# (?!\d): negative lookahead - không có digit đứng sau
STANDALONE_TIMESTAMP_RE = re.compile(r"(?<!\d)\d{1,2}:\d{2}(?::\d{2})?(?!\d)")

# Pattern 5: Xóa filler words đầu câu (uh, um, okay, well, so, yeah)
# ^: đầu chuỗi
# (?:...)+: một hoặc nhiều filler
# \b: word boundary
# [\s,.-]*: khoảng trắng/punctuation đứng sau filler
LEADING_FILLER_RE = re.compile(
    r"^(?:(?:uh|um|erm|er|ah|hmm|mm|okay|ok|well|so|yeah)\b[\s,.-]*)+",
    re.IGNORECASE,
)

# Pattern 6: Xóa filler words cuối câu
TRAILING_FILLER_RE = re.compile(
    r"(?:[\s,.;!?-]*(?:uh|um|erm|er|ah|hmm|mm|you know|i mean))+$",
    re.IGNORECASE,
)

# Pattern 7: Xóa filler words trong câu (giữ lại punctuation)
# Danh sách các pattern để xử lý filler nội dung
INLINE_FILLER_PATTERNS = [
    # Xóa "uh", "um", "er", etc. trong câu
    re.compile(r"(?i)(^|[\s,.;!?-])(?:uh|um|erm|er|ah|hmm|mm)(?=$|[\s,.;!?-])"),
    # Xóa "you know", "i mean" trong câu
    re.compile(r"(?i)(^|[\s,.;!?-])(?:you know|i mean)(?=$|[\s,.;!?-])"),
]


def clean_segment_text(text: str) -> str:
    """
    Làm sạch một segment text.

    Args:
        text: Text đầu vào từ transcript

    Returns:
        Text đã làm sạch
    """
    # =============================================================================
    # BƯỚC 1: Xử lý newline/CRLF
    # =============================================================================
    # Thay \r\n bằng khoảng trắng đơn
    if not text:
        return ""

    text = text.replace("\r", " ").replace("\n", " ")

    # =============================================================================
    # BƯỚC 2: Xóa timestamp (nếu có trong text)
    # =============================================================================
    text = TIMESTAMP_RANGE_RE.sub(" ", text)
    text = STANDALONE_TIMESTAMP_RE.sub(" ", text)

    # =============================================================================
    # BƯỚC 3: Xóa marker noise và HTML
    # =============================================================================
    text = BRACKET_NOISE_RE.sub(" ", text)
    text = HTML_TAG_RE.sub(" ", text)

    # =============================================================================
    # BƯỚC 4: Chuẩn hóa whitespace (nhiều space -> 1 space)
    # =============================================================================
    text = re.sub(r"\s+", " ", text).strip()

    # =============================================================================
    # BƯỚC 5: Xóa filler đầu câu
    # =============================================================================
    text = LEADING_FILLER_RE.sub("", text)

    # =============================================================================
    # BƯỚC 6: Xóa filler trong câu (giữ lại dấu phân cách)
    # =============================================================================
    for pattern in INLINE_FILLER_PATTERNS:
        # Lambda giữ lại ký tự đứng trước filler (dấu phân cách)
        text = pattern.sub(lambda m: m.group(1), text)

    # =============================================================================
    # BƯỚC 7: Xóa filler cuối câu
    # =============================================================================
    text = TRAILING_FILLER_RE.sub("", text)

    # =============================================================================
    # BƯỚC 8: Chuẩn hóa dấu câu và whitespace
    # =============================================================================
    # Xóa space trước dấu câu: "text , " -> "text,"
    text = re.sub(r"\s+([,.;!?])", r"\1", text)
    # Xóa dấu lặp: "!!!" -> "!"
    text = re.sub(r"([,.;!?]){2,}", r"\1", text)
    # Xóa empty parentheses: "() "
    text = re.sub(r"\(\s*\)", " ", text)
    # Xóa nhiều space liên tiếp
    text = re.sub(r"\s{2,}", " ", text)

    # =============================================================================
    # BƯỚC 9: Xóa punctuation đầu/cuối
    # =============================================================================
    return text.strip(" -.,;:!?")


def clean_record_preserve_structure(record: dict) -> dict:
    """
    Làm sạch một record transcript nhưng giữ nguyên cấu trúc.

    Args:
        Record có video_id, title, transcript: [{text, start, duration}, ...]

    Returns:
        Record đã làm sạch với cấu trúc tương tự
    """
    # Copy dict gốc để không modify original
    cleaned = dict(record)
    cleaned_segments = []

    # Duyệt qua từng segment trong transcript
    for seg in record.get("transcript", []):
        # Lấy text và clean
        cleaned_text = clean_segment_text(seg.get("text", ""))

        # Bỏ qua segment rỗng sau khi clean
        if not cleaned_text:
            continue

        # Thêm segment đã clean vào danh sách
        cleaned_segments.append(
            {
                "text": cleaned_text,
                "start": seg.get("start"),
                "duration": seg.get("duration"),
            }
        )

    # Cập nhật transcript đã clean
    cleaned["transcript"] = cleaned_segments
    return cleaned


def clean_transcripts_file(input_path: Path, output_path: Path) -> None:
    """
    Xử lý toàn bộ file transcript.

    Args:
        input_path: File input .jsonl
        output_path: File output .jsonl
    """
    total_records = 0  # Số video đã xử lý
    total_segments_before = 0  # Tổng segments trước clean
    total_segments_after = 0  # Tổng segments sau clean

    # =============================================================================
    # ĐỌC VÀ GHI FILE
    # =============================================================================
    with (
        input_path.open("r", encoding="utf-8") as fin,
        output_path.open("w", encoding="utf-8") as fout,
    ):
        # Duyệt từng dòng trong file
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            # Parse JSON
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"SKIP line {line_no}: invalid JSON")
                continue

            # Đếm segments trước clean
            total_records += 1
            total_segments_before += len(record.get("transcript", []))

            # Clean record
            cleaned_record = clean_record_preserve_structure(record)
            total_segments_after += len(cleaned_record.get("transcript", []))

            # Ghi ra file output
            fout.write(json.dumps(cleaned_record, ensure_ascii=False) + "\n")

    # In thống kê
    print(f"Done: {total_records} records")
    print(f"Segments before: {total_segments_before}")
    print(f"Segments after : {total_segments_after}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    clean_transcripts_file(INPUT_PATH, OUTPUT_PATH)
