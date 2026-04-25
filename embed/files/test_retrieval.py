"""
tests/test_retrieval.py
Unit tests cho Member 2 pipeline.

Chạy:
    pytest tests/ -v
    pytest tests/ -v -k "test_tokenize"    # chạy 1 test cụ thể
"""

import sys
import json
import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Import các module cần test ───────────────────────────────────────────────
from scripts.s02_hybrid_search import (
    tokenize,
    build_bm25,
    min_max_normalize,
    reciprocal_rank_fusion,
    HybridSearcher,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Tokenizer
# ═══════════════════════════════════════════════════════════════════════════════
class TestTokenize:
    def test_basic(self):
        assert tokenize("Hello World") == ["hello", "world"]

    def test_hyphen_preserved(self):
        tokens = tokenize("self-attention fine-tuning")
        assert "self-attention" in tokens
        assert "fine-tuning" in tokens

    def test_numbers(self):
        tokens = tokenize("top-5 accuracy 98.5%")
        assert "top-5" in tokens
        assert "98" in tokens or "98" in " ".join(tokens)

    def test_empty_string(self):
        assert tokenize("") == []

    def test_special_chars_stripped(self):
        tokens = tokenize("BLEU score: 0.42!")
        assert "bleu" in tokens
        assert "score" in tokens


# ═══════════════════════════════════════════════════════════════════════════════
# 2. BM25
# ═══════════════════════════════════════════════════════════════════════════════
SAMPLE_CORPUS = [
    "The transformer model uses self-attention mechanisms.",
    "Gradient descent optimizes the loss function iteratively.",
    "Backpropagation computes gradients through the network.",
    "BLEU score measures machine translation quality.",
    "Dropout is a regularization technique for neural networks.",
]


class TestBM25:
    @pytest.fixture
    def bm25(self):
        return build_bm25(SAMPLE_CORPUS)

    def test_build_succeeds(self, bm25):
        assert bm25 is not None

    def test_relevant_doc_scores_higher(self, bm25):
        """Query 'transformer attention' nên rank doc 0 cao nhất."""
        from rank_bm25 import BM25Okapi
        scores = bm25.get_scores(tokenize("transformer attention"))
        assert np.argmax(scores) == 0

    def test_all_docs_scored(self, bm25):
        scores = bm25.get_scores(tokenize("neural network"))
        assert len(scores) == len(SAMPLE_CORPUS)

    def test_oov_query_no_crash(self, bm25):
        """Query hoàn toàn OOV không crash."""
        scores = bm25.get_scores(["xyzabc123"])
        assert all(s == 0.0 for s in scores)

    def test_persist_and_load(self, bm25, tmp_path):
        bm25_path = tmp_path / "bm25.pkl"
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25, f)
        with open(bm25_path, "rb") as f:
            loaded = pickle.load(f)
        scores_orig   = bm25.get_scores(tokenize("gradient"))
        scores_loaded = loaded.get_scores(tokenize("gradient"))
        np.testing.assert_array_almost_equal(scores_orig, scores_loaded)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Score normalization & RRF
# ═══════════════════════════════════════════════════════════════════════════════
class TestScoreUtils:
    def test_min_max_basic(self):
        scores = np.array([1.0, 3.0, 2.0])
        normed = min_max_normalize(scores)
        assert normed.min() == pytest.approx(0.0)
        assert normed.max() == pytest.approx(1.0)

    def test_min_max_all_equal(self):
        scores = np.array([5.0, 5.0, 5.0])
        normed = min_max_normalize(scores)
        assert all(n == 0.0 for n in normed)

    def test_rrf_union(self):
        """RRF phải gom docs từ cả 2 list."""
        list1 = [0, 1, 2]
        list2 = [3, 4, 5]
        rrf = reciprocal_rank_fusion([list1, list2])
        assert set(rrf.keys()) == {0, 1, 2, 3, 4, 5}

    def test_rrf_top_ranked_higher(self):
        """Doc ở rank 1 của cả 2 list nên có score cao nhất."""
        list1 = [99, 1, 2]
        list2 = [99, 3, 4]
        rrf = reciprocal_rank_fusion([list1, list2])
        # Doc 99 xuất hiện đầu cả 2 list → score cao nhất
        assert max(rrf, key=rrf.get) == 99


# ═══════════════════════════════════════════════════════════════════════════════
# 4. HybridSearcher (mock FAISS + model)
# ═══════════════════════════════════════════════════════════════════════════════
def _make_mock_searcher(n_docs=5, fusion="linear"):
    """Tạo HybridSearcher với FAISS index và model giả."""
    import faiss

    # Metadata giả
    meta = pd.DataFrame({
        "chunk_idx":  list(range(n_docs)),
        "video_id":   [f"vid_{i}" for i in range(n_docs)],
        "title":      [f"Lecture {i}" for i in range(n_docs)],
        "course":     ["CS25_Transformers", "CS229_ML", "CS224N_NLP",
                       "CS224R_RL", "CME296_Diffusion"][:n_docs],
        "start_time": [float(i * 60) for i in range(n_docs)],
        "end_time":   [float(i * 60 + 50) for i in range(n_docs)],
        "duration":   [50.0] * n_docs,
        "word_count": [120] * n_docs,
        "chunk_index": list(range(n_docs)),
        "playlist_id": ["PL_test"] * n_docs,
        "published_at": ["2023-01-01"] * n_docs,
        "chunk_text":  [
            "The transformer model uses self-attention mechanisms.",
            "Gradient descent optimizes the loss function iteratively.",
            "BLEU score measures machine translation quality.",
            "Policy gradient in reinforcement learning.",
            "Diffusion model forward and reverse process.",
        ][:n_docs],
        "url": [f"https://youtu.be/vid_{i}?t={i*60}" for i in range(n_docs)],
    })

    # FAISS với random vectors
    dim = 16
    vecs = np.random.rand(n_docs, dim).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs /= norms
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    # BM25
    bm25 = build_bm25(meta["chunk_text"].tolist())

    # Mock model: selalu return random vector normalized
    mock_model = MagicMock()
    mock_model.encode.return_value = (vecs[0:1])  # shape (1, dim)

    return HybridSearcher(index, bm25, meta, mock_model, fusion=fusion)


class TestHybridSearcher:
    def test_search_returns_dataframe(self):
        searcher = _make_mock_searcher()
        results = searcher.search("attention mechanism", top_k=3)
        assert isinstance(results, pd.DataFrame)

    def test_search_top_k_respected(self):
        searcher = _make_mock_searcher(n_docs=5)
        results = searcher.search("transformer", top_k=3)
        assert len(results) <= 3

    def test_results_have_required_columns(self):
        searcher = _make_mock_searcher()
        results = searcher.search("gradient descent", top_k=2)
        for col in ["rank", "score", "title", "course", "url", "snippet"]:
            assert col in results.columns, f"Missing column: {col}"

    def test_scores_are_non_negative(self):
        searcher = _make_mock_searcher()
        results = searcher.search("neural network", top_k=3)
        assert all(results["score"] >= 0), "Scores không được âm"

    def test_ranks_are_sequential(self):
        searcher = _make_mock_searcher()
        results = searcher.search("self-attention", top_k=4)
        ranks = results["rank"].tolist()
        assert ranks == list(range(1, len(ranks) + 1))

    def test_rrf_fusion_works(self):
        searcher = _make_mock_searcher(fusion="rrf")
        results = searcher.search("policy gradient", top_k=3)
        assert len(results) > 0

    def test_url_format(self):
        searcher = _make_mock_searcher()
        results = searcher.search("transformer", top_k=1)
        url = results.iloc[0]["url"]
        assert url.startswith("https://youtu.be/")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Queries JSON
# ═══════════════════════════════════════════════════════════════════════════════
class TestSampleQueries:
    QUERIES_FILE = Path(__file__).parent.parent / "data" / "sample_queries" / "vn_queries_20.json"

    def test_file_exists(self):
        assert self.QUERIES_FILE.exists(), f"Missing: {self.QUERIES_FILE}"

    def test_exactly_20_queries(self):
        with open(self.QUERIES_FILE) as f:
            queries = json.load(f)
        assert len(queries) == 20

    def test_all_queries_have_required_fields(self):
        with open(self.QUERIES_FILE) as f:
            queries = json.load(f)
        for q in queries:
            assert "id"              in q, f"Missing 'id' in query {q}"
            assert "query_vi"        in q, f"Missing 'query_vi' in query {q}"
            assert "expected_course" in q, f"Missing 'expected_course' in query {q}"

    def test_all_queries_non_empty(self):
        with open(self.QUERIES_FILE) as f:
            queries = json.load(f)
        for q in queries:
            assert len(q["query_vi"].strip()) > 5, f"Query quá ngắn: {q}"

    def test_ids_are_unique_and_sequential(self):
        with open(self.QUERIES_FILE) as f:
            queries = json.load(f)
        ids = [q["id"] for q in queries]
        assert len(set(ids)) == 20
        assert sorted(ids) == list(range(1, 21))
