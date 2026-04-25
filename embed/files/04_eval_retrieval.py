"""
04_eval_retrieval.py
Ngày 7: Đánh giá chất lượng retrieval trên 20 queries tiếng Việt mẫu.

Metrics:
  - Hit@K: tỉ lệ query có ít nhất 1 kết quả đúng course trong top-K
  - MRR@K: Mean Reciprocal Rank
  - Avg latency: thời gian trung bình per query (ms)

Output: bảng kết quả per-query + summary metrics.

Chạy:
    python scripts/04_eval_retrieval.py
    python scripts/04_eval_retrieval.py --top-k 3 --fusion rrf
    python scripts/04_eval_retrieval.py --verbose          # in chi tiết kết quả
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from scripts.s02_hybrid_search import load_searcher

QUERIES_PATH = cfg.DATA_DIR / "sample_queries" / "vn_queries_20.json"


def load_queries(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate(
    searcher,
    queries: list[dict],
    top_k: int,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Chạy từng query, tính hit và reciprocal rank.
    """
    rows = []
    for q in queries:
        query_text   = q["query_vi"]
        expected     = q["expected_course"]

        t0 = time.time()
        results = searcher.search(query_text, top_k=top_k)
        latency_ms = (time.time() - t0) * 1000

        # Hit: top-K kết quả có ít nhất 1 đúng course không?
        hit = int(any(r["course"] == expected for _, r in results.iterrows()))

        # Reciprocal Rank
        rr = 0.0
        for _, r in results.iterrows():
            if r["course"] == expected:
                rr = 1.0 / r["rank"]
                break

        if verbose:
            print(f"\n{'─'*60}")
            print(f"Q{q['id']:02d}: {query_text}")
            print(f"     Expected: {expected} | Hit={hit} | RR={rr:.3f}")
            for _, r in results.iterrows():
                mark = "✅" if r["course"] == expected else "  "
                print(f"  {mark} [{r['rank']}] {r['course']} | {r['title'][:55]}")
                print(f"        {r['url']}")

        rows.append({
            "id":         q["id"],
            "query":      query_text[:50] + "...",
            "expected":   expected,
            "hit":        hit,
            "rr":         rr,
            "latency_ms": round(latency_ms, 1),
            "top1_course": results.iloc[0]["course"] if len(results) > 0 else "—",
        })

    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame, top_k: int) -> None:
    hit_rate = df["hit"].mean() * 100
    mrr      = df["rr"].mean()
    avg_lat  = df["latency_ms"].mean()

    print(f"\n{'═'*55}")
    print(f"  📊 Evaluation Summary  (top_k={top_k}, n={len(df)} queries)")
    print(f"{'═'*55}")
    print(f"  Hit@{top_k}       : {hit_rate:.1f}%  ({df['hit'].sum()}/{len(df)} queries)")
    print(f"  MRR@{top_k}       : {mrr:.4f}")
    print(f"  Avg latency : {avg_lat:.0f} ms / query")
    print(f"{'═'*55}\n")

    # Per-course breakdown
    print("  Per-course Hit@K:")
    course_stats = df.groupby("expected")["hit"].agg(["sum", "count"])
    course_stats["hit_rate"] = (course_stats["sum"] / course_stats["count"] * 100).round(1)
    print(course_stats[["sum", "count", "hit_rate"]].rename(
        columns={"sum": "hits", "count": "total", "hit_rate": "hit%"}
    ).to_string())
    print()


def main():
    parser = argparse.ArgumentParser(description="Eval retrieval on 20 VN queries")
    parser.add_argument("--top-k",   type=int, default=cfg.TOP_K_FINAL)
    parser.add_argument("--alpha",   type=float, default=cfg.HYBRID_ALPHA)
    parser.add_argument("--fusion",  type=str, default="linear",
                        choices=["linear", "rrf"])
    parser.add_argument("--verbose", action="store_true",
                        help="In chi tiết từng query")
    parser.add_argument("--output",  type=Path, default=None,
                        help="Lưu kết quả ra CSV (optional)")
    args = parser.parse_args()

    if not QUERIES_PATH.exists():
        sys.exit(f"[error] Không tìm thấy: {QUERIES_PATH}")

    queries  = load_queries(QUERIES_PATH)
    searcher = load_searcher(alpha=args.alpha, fusion=args.fusion)

    print(f"\n[eval] Chạy {len(queries)} queries | top_k={args.top_k} | fusion={args.fusion}")
    df = evaluate(searcher, queries, top_k=args.top_k, verbose=args.verbose)
    print_summary(df, top_k=args.top_k)

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"[eval] Kết quả đã lưu → {args.output}")


if __name__ == "__main__":
    main()
