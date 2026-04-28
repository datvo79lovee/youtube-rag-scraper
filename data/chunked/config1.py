"""
Cau hinh cho Pipeline 2 - Cross-lingual Knowledge Units
==================================================
Tac dung:
  - Dinh nghia cac thong so cau hinh cho Pipeline 2
  - Su dung frozen dataclass de dam bao tinh bat bien (immutable)

Cac tham so:
  - Chunk settings: token range, duration
  - Topic/Keywords: so luong, do dai toi thieu
  - Cross-lingual: bat/tat Vietnamese hints
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Pipeline2Config:
    # DUONG DAN FILE
    root_dir: Path  # Thu muc goc project
    input_path: Path  # File input: transcript_v3.jsonl
    output_path: Path  # File output: transcript_v4.jsonl

    # CAU HINH CHUNKING
    # So token toi thieu trong mot chunk
    min_tokens: int = 150
    # So token toi da trong mot chunk
    max_tokens: int = 200
    # Thoi gian toi da cua chunk (giay)
    max_duration: float = 90.0
    # Thoi gian toi thieu de chunk hop ly
    min_duration: float = 20.0
    # Buffer cho phep vuot qua khi cau chua hoan thanh
    sentence_buffer: int = 20

    # TOPIC/KEYWORDS
    # So tu khoa toi da
    max_keywords: int = 8
    # Tu khoa toi thieu bao nhieu ky tu
    min_keyword_length: int = 3

    # CROSS-LINGUAL
    # Bat/tat Vietnamese keywords hint
    enable_vi_hints: bool = True

    # OUTPUT STRUCTURE
    # Cac truong se duoc ghi vao output
    output_fields: tuple = (
        "video_id",
        "title",
        "course",
        "source",
        "chunk_type",
        "chunk_text",
        "embedding_text",
        "start_time",
        "end_time",
        "duration",
        "token_count",
    )


def build_pipeline2_config() -> Pipeline2Config:
    """
    Tao cau hinh mac dinh cho Pipeline 2
    Tu dong tim thu muc root cua project

    Returns:
        Pipeline2Config: Cau hinh voi cac gia tri mac dinh
    """
    root = Path(__file__).resolve().parent.parent.parent
    return Pipeline2Config(
        root_dir=root,
        input_path=root / "data" / "chunked" / "transcript_v3.jsonl",
        output_path=root / "data" / "chunked" / "transcript_v4.jsonl",
    )
