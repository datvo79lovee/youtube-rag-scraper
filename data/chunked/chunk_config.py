from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ChunkConfig:
    root_dir: Path
    input_path: Path
    output_dir: Path
    strategy: str = "hybrid"

    target_tokens: int = 150
    min_chunk_words: int = 40
    max_chunk_words: int = 140

    max_chunk_duration: float = 60.0
    max_silence_gap: float = 8.0
    overlap_segments: int = 2

    max_semantic_words: int = 220
    max_semantic_duration: float = 120.0
    min_lexical_overlap: float = 0.08

    semantic_similarity_threshold: float = 0.75
    max_embedding_tokens: int = 300
    min_chunk_chars: int = 80
    
    chunk_version: str = "v1"
    write_intermediate_time_chunks: bool = True

    @property
    def time_output_path(self) -> Path:
        return self.output_dir / "transcripts_time_chunked.jsonl"

    @property
    def semantic_output_path(self) -> Path:
        return self.output_dir / "transcripts_semantic_chunked.jsonl"

    @property
    def hybrid_output_path(self) -> Path:
        return self.output_dir / "transcripts_hybrid_chunked.jsonl"


def build_default_config(strategy: str = "hybrid") -> ChunkConfig:
    root_dir = Path(__file__).resolve().parent.parent.parent
    return ChunkConfig(
        root_dir=root_dir,
        input_path = root_dir / "data" / "cleaned" / "transcripts_clean_sentence.jsonl",
        output_dir=root_dir / "data" / "chunked",
        strategy=strategy.lower().strip(),
    )
