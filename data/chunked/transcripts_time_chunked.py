"""
transcripts_time_chunked.py - Chia transcript theo thời gian cố định
======================================================
Mục đích: Chia transcript thành các chunk có kích thước cố định
           theo số từ và thời gian

Pipeline: data/cleaned/transcripts_clean.jsonl -> data/chunked/transcripts_time_chunked.jsonl

Nguyên tắc chia chunk:
- MIN_CHUNK_WORDS:ít nhất 40 từ -> tránh chunk quá ngắn
- MAX_CHUNK_WORDS:tối đa 140 từ -> fit context window của LLM
- MAX_CHUNK_DURATION: tối đa 60 giây -> nghe liên tục không quá lâu
- MAX_SILENCE_GAP: gap > 8 giây -> xác định điểm ngắt tự nhiên
- Kết thúc câu (., !, ?) -> ngắt chunk

Sau khi chia xong, gộp các chunk quá ngắn (< 20 từ) vào chunk trước đó.
"""

import json
import re
from pathlib import Path
from typing import Iterable

# CẤU HÌNH ĐƯỜNG DẪN
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_PATH = ROOT_DIR / "data" / "cleaned" / "transcripts_clean.jsonl"
OUTPUT_PATH = ROOT_DIR / "data" / "chunked" / "transcripts_time_chunked.jsonl"

# CẤU HÌNH CHIA CHUNK
# Số từ tối thiểu/trong một chunk
MIN_CHUNK_WORDS = 40
# Số từ tối đa trong một chunk (fit trong context LLM)
MAX_CHUNK_WORDS = 140
# Thời gian tối đa của một chunk (giây)
MAX_CHUNK_DURATION = 60.0
# Khoảng im lặng tối đa để ngắt chunk (giây)
MAX_SILENCE_GAP = 8.0
# Số từ tối thiểu để xem segment là "ngắn"
SHORT_SEGMENT_WORDS = 4


def normalize_chunk_text(text: str) -> str:
    """
    Chuẩn hóa text của chunk.

    Args:
        text: Text đầu vào

    Returns:
        Text đã chuẩn hóa với:
        - \\n -> space
        - Nhiều space -> 1 space
        - Space trước dấu câu -> bỏ
        - Dấu câu lặp -> 1 dấu
        - Viết hoa chữ cái đầu
    """
    # BƯỚC 1: Xử lý newline
    text = text.replace("\n", " ").replace("\r", " ")

    # BƯỚC 2: Chuẩn hóa whitespace
    text = re.sub(r"\s+", " ", text)
    # BƯỚC 3: Xóa space trước dấu câu
    text = re.sub(r"\s+([,.;!?])", r"\1", text)
    # BƯỚC 4: Xóa dấu câu lặp
    text = re.sub(r"([,.;!?]){2,}", r"\1", text)
    # BƯỚC 5: Xóa punctuation đầu/cuối
    text = text.strip(" -.,;:!?")
    if not text:
        return ""
    # BƯỚC 6: Viết hoa chữ cái đầu
    return text[0].upper() + text[1:]


def word_count(text: str) -> int:
    """
    Đếm số từ trong text.

    Args:
        text: Chuỗi cần đếm

    Returns:
        Số từ (tách bằng space)
    """
    return len(text.split())


def segment_end(seg: dict) -> float:
    """
    Tính thời điểm kết thúc của segment.

    Args:
        segment: Dict có start và duration

    Returns:
        Thời điểm kết thúc (start + duration)
    """
    return round(float(seg["start"]) + float(seg["duration"]), 3)


def should_close_chunk(
    current_segments: list[dict],
    next_seg: dict | None,
    current_words: int,
    chunk_start: float,
    current_end: float,
) -> bool:
    """
    Quyết định có nên ngắt chunk hiện tại hay không.

    Logic quyết định (OR - bất kỳ điều kiện nào đúng đều ngắt):
    1. Đã đạt MAX_CHUNK_WORDS (140 từ)
    2. Đạt MAX_CHUNK_DURATION (60s) VÀ có đủ MIN_CHUNK_WORDS (40 từ)
    3. Là segment cuối cùng
    4. Khoảng im lặng > MAX_SILENCE_GAP (8s) VÀ có đủ từ
    5. Text kết thúc bằng dấu câu (. ! ?) VÀ có đủ từ

    Args:
        current_segments: Các segment trong chunk hiện tại
        next_seg: Segment tiếp theo (None nếu là cuối)
        current_words: Tổng số từ trong chunk
        chunk_start: Thời điểm bắt đầu chunk
        current_end: Thời điểm kết thúc segment cuối

    Returns:
        True nên ngắt chunk, False nên tiếp tục
    """
    # Tính thời gian chunk đã chạy
    duration = current_end - chunk_start

    # ĐIỀU KIỆN 1: Quá số từ tối đa
    if current_words >= MAX_CHUNK_WORDS:
        return True

    # ĐIỀU KIỆN 2: Quá thời gian tối đa VÀ có đủ từ
    if duration >= MAX_CHUNK_DURATION and current_words >= MIN_CHUNK_WORDS:
        return True

    # ĐIỀU KIỆN 3: Là segment cuối cùng
    if next_seg is None:
        return True

    # ĐIỀU KIỆN 4: Khoảng im lặng lớn
    next_gap = float(next_seg["start"]) - current_end
    if next_gap > MAX_SILENCE_GAP and current_words >= MIN_CHUNK_WORDS:
        return True

    # ĐIỀU KIỆN 5: Kết thúc câu hoàn chỉnh
    current_text = current_segments[-1]["text"].strip()
    if current_words >= MIN_CHUNK_WORDS and re.search(r"[.!?]$", current_text):
        return True

    return False


def merge_short_tail(chunks: list[dict]) -> list[dict]:
    """
    Gộp các chunk quá ngắn vào chunk trước đó.

    Giải thích: Sau khi chia chunk, có thể có vài chunk rất ngắn
    (ví dụ: chỉ 5-10 từ). Ta gộp chúng vào chunk trước đó
    để tạo chunk hoàn chỉnh hơn.

    Args:
        chunks: Danh sách chunks đã chia

    Returns:
        Danh sách chunks đã gộp
    """
    if len(chunks) < 2:
        return chunks

    merged = []
    for chunk in chunks:
        # Nếu chunk hiện tại < 20 từ (MIN_CHUNK_WORDS // 2)
        # -> gộp vào chunk trước đó
        if merged and chunk["word_count"] < MIN_CHUNK_WORDS // 2:
            prev = merged[-1]
            # Nối text
            prev["chunk_text"] = normalize_chunk_text(
                prev["chunk_text"] + " " + chunk["chunk_text"]
            )
            # Cập nhật end_time
            prev["end_time"] = chunk["end_time"]
            # Tính lại duration
            prev["duration"] = round(prev["end_time"] - prev["start_time"], 3)
            # Tính lại word_count
            prev["word_count"] = word_count(prev["chunk_text"])
            # Cộng segment count
            prev["segment_count"] += chunk["segment_count"]
            # Cập nhật segment index
            prev["source_segment_end_idx"] = chunk["source_segment_end_idx"]
        else:
            # Giữ nguyên chunk
            merged.append(chunk)

    return merged


def build_time_chunks(record: dict) -> list[dict]:
    """
    Chia một transcript video thành các time chunks.

    Args:
        record: Dict có video_id, title, transcript: [{text, start, duration}, ...]

    Returns:
        Danh sách các chunk đã chia
    """
    # Lấy transcript
    transcript = record.get("transcript", [])
    if not transcript:
        return []

    chunks = []
    current_segments = []  # Segment đang trong chunk hiện tại
    current_words = 0  # Số từ trong chunk hiện tại
    chunk_start = None  # Thời điểm bắt đầu chunk

    # DUYỆT QUA TỪNG SEGMENT
    for idx, seg in enumerate(transcript):
        # Normalize và lấy text
        text = normalize_chunk_text(seg.get("text", ""))
        if not text:
            continue

        # Tạo segment object
        seg_obj = {
            "text": text,
            "start": float(seg["start"]),
            "duration": float(seg["duration"]),
            "end": segment_end(seg),
            "segment_idx": idx,
        }

        # Nếu là segment đầu tiên -> bắt đầu chunk mới
        if chunk_start is None:
            chunk_start = seg_obj["start"]

        # Thêm vào danh sách segment hiện tại
        current_segments.append(seg_obj)
        current_words += word_count(text)
        current_end = seg_obj["end"]

        # Lấy segment tiếp theo (nếu có)
        next_seg = transcript[idx + 1] if idx + 1 < len(transcript) else None

        # Kiểm tra có nên ngắt chunk không
        if should_close_chunk(
            current_segments=current_segments,
            next_seg=next_seg,
            current_words=current_words,
            chunk_start=chunk_start,
            current_end=current_end,
        ):
            # Nối tất cả segment thành chunk_text
            chunk_text = normalize_chunk_text(
                " ".join(segment["text"] for segment in current_segments)
            )
            if chunk_text:
                # Tạo chunk record
                chunks.append(
                    {
                        "video_id": record["video_id"],
                        "title": record.get("title", ""),
                        "course": record.get("course", ""),
                        "playlist_id": record.get("playlist_id", ""),
                        "published_at": record.get("published_at", ""),
                        "source": record.get("source", ""),
                        "chunk_type": "time",
                        "chunk_index": len(chunks),
                        "chunk_text": chunk_text,
                        "start_time": round(chunk_start, 3),
                        "end_time": round(current_end, 3),
                        "duration": round(current_end - chunk_start, 3),
                        "word_count": word_count(chunk_text),
                        "segment_count": len(current_segments),
                        "source_segment_start_idx": current_segments[0]["segment_idx"],
                        "source_segment_end_idx": current_segments[-1]["segment_idx"],
                    }
                )

            # Reset cho chunk tiếp theo
            current_segments = []
            current_words = 0
            chunk_start = None

    # GỘP CÁC CHUNK NGẮN
    return merge_short_tail(chunks)


def iter_jsonl(path: Path) -> Iterable[dict]:
    """
    Đọc file JSONL dạng generator (tiết kiệm memory).

    Args:
        path: Đường dẫn file .jsonl

    Yields:
        Mỗi record JSON trong file
    """
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"SKIP line {line_no}: invalid JSON")


def write_jsonl(records: Iterable[dict], path: Path) -> None:
    """
    Ghi danh sách records ra file JSONL.

    Args:
        records: Danh sách records
        path: Đường dẫn file output
    """
    # Tạo thư mục nếu chưa có
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    print(f"Wrote {count} chunks -> {path}")


def main() -> None:
    """Hàm chính - xử lý tất cả videos."""
    output_records = []
    total_videos = 0

    # Đọc từng video và chia chunk
    for record in iter_jsonl(INPUT_PATH):
        total_videos += 1
        output_records.extend(build_time_chunks(record))

    # Ghi output
    write_jsonl(output_records, OUTPUT_PATH)
    print(f"Processed videos: {total_videos}")
    print(f"Total time chunks: {len(output_records)}")


if __name__ == "__main__":
    main()
