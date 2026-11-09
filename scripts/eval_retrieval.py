"""Retrieval sanity check: do patient-style questions find the right knowledge card?

Usage:  python scripts/eval_retrieval.py                 (tfidf baseline, offline)
        python scripts/eval_retrieval.py --embedder both  (also MiniLM; needs: pip install sentence-transformers)

retrieval_eval.json (30 queries) was used while building the cards, so it is in-sample. retrieval_eval_heldout.json (14 queries) was written afterwards
and run once. All queries are hand-written by the project author: this is a sanity check, not a benchmark.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.cards import DATA_DIR, load_cards            # noqa: E402
from src.rag.embedders import MiniLMEmbedder, TfidfEmbedder  # noqa: E402
from src.rag.retriever import Retriever                    # noqa: E402


def evaluate(retriever: Retriever, queries: list[dict], ks=(1, 3)) -> dict:
    hits = {k: 0 for k in ks}
    rr, misses = 0.0, []
    for item in queries:
        ranked = [c.id for c, _ in retriever.search(item["query"], k=len(retriever.cards))]
        rank = next((i + 1 for i, cid in enumerate(ranked) if cid in item["relevant"]), None)
        for k in ks:
            hits[k] += rank is not None and rank <= k
        rr += 1.0 / rank if rank else 0.0
        if rank is None or rank > 3:
            misses.append((item["query"], item["relevant"], ranked[:3]))
    n = len(queries)
    return {**{f"recall@{k}": hits[k] / n for k in ks}, "mrr": rr / n, "n": n, "misses": misses}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", choices=["tfidf", "minilm", "both"], default="tfidf")
    args = ap.parse_args()
    cards = load_cards()
    makers = {"tfidf": TfidfEmbedder, "minilm": MiniLMEmbedder}
    for name in (["tfidf", "minilm"] if args.embedder == "both" else [args.embedder]):
        retriever = Retriever(cards, makers[name]())
        for f in sorted(DATA_DIR.glob("retrieval_eval*.json")):
            queries = json.loads(f.read_text(encoding="utf-8"))
            res = evaluate(retriever, queries)
            print(f"{name:7s} {f.stem:24s} recall@1 {res['recall@1']:.3f} | recall@3 {res['recall@3']:.3f} | MRR {res['mrr']:.3f} | n={res['n']}")
            for q, rel, top in res["misses"]:
                print(f"   missed outside top-3: {q!r} expected {rel} got {top}")
