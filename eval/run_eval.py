"""Offline evaluation of drafted replies on a small sample ticket set.

For each sample ticket the full workflow is run (with the deterministic
offline LLM/embedder) up to the approval gate, and the draft is scored
on three checks:

1. the predicted category matches the expected category;
2. the expected help-centre document is among the cited sources;
3. every expected key point appears in the drafted reply.

Usage (from the repository root):
    python eval/run_eval.py

These are sample-set results on six hand-written tickets — a smoke
check for the pipeline, not a production quality metric.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.embeddings import HashingEmbedder
from app.llm import RuleBasedLLM
from app.pipeline import SupportDesk


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "sample_tickets.json"), encoding="utf-8") as fh:
        samples = json.load(fh)

    desk = SupportDesk(llm=RuleBasedLLM(), embedder=HashingEmbedder())

    passed = 0
    category_hits = 0
    citation_hits = 0
    keypoint_hits = 0

    for i, sample in enumerate(samples, start=1):
        record = desk.submit(sample["subject"], sample["body"])
        draft = record["draft_response"].lower()

        category_ok = record["category"] == sample["expected_category"]
        citation_ok = sample["expected_doc"] in record["citations"]
        keypoints_ok = all(
            point.lower() in draft for point in sample["key_points"]
        )
        all_ok = category_ok and citation_ok and keypoints_ok

        passed += all_ok
        category_hits += category_ok
        citation_hits += citation_ok
        keypoint_hits += keypoints_ok

        print(f"[{i}] {sample['subject']}")
        print(f"    category: {'ok' if category_ok else 'MISS'} "
              f"(got {record['category']}, expected {sample['expected_category']})")
        print(f"    citation: {'ok' if citation_ok else 'MISS'} "
              f"(expected {sample['expected_doc']} in {record['citations']})")
        print(f"    key points: {'ok' if keypoints_ok else 'MISS'} "
              f"(expected {sample['key_points']})")
        print(f"    => {'PASS' if all_ok else 'FAIL'}")

    total = len(samples)
    print()
    print(f"Summary over {total} sample tickets:")
    print(f"  category correct : {category_hits}/{total}")
    print(f"  citation correct : {citation_hits}/{total}")
    print(f"  key points found : {keypoint_hits}/{total}")
    print(f"  all checks passed: {passed}/{total}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
