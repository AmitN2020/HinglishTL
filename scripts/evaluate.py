"""Run the committed benchmark against a live assistant service.

Usage: SERVICE_BASE_URL=https://... python scripts/evaluate.py
Writes artifacts/evaluation_predictions.json and artifacts/evaluation_report.json.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from hinglish_assistant.client import AssistantClient  # noqa: E402
from hinglish_assistant.metrics import aggregate_transliteration, corpus_bleu  # noqa: E402

ROOT = Path(__file__).parents[1]


def main() -> None:
    cases = json.loads((ROOT / "data" / "evaluation_cases.json").read_text(encoding="utf-8"))
    client = AssistantClient.from_environment()
    predictions = []
    for case in cases:
        result = client.assist(case["input"], rewrite=True, translate=True)
        predictions.append({**case, **result.model_dump()})

    raw_hindi = [row["transliteration"] for row in predictions]
    reference_hindi = [row["reference_devanagari"] for row in predictions]
    references = [row["reference_english"] for row in predictions]
    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in predictions:
        by_category[row["category"]].append(row)

    report = {
        "case_count": len(predictions),
        "transliteration": aggregate_transliteration(raw_hindi, reference_hindi).__dict__,
        "english_bleu": {
            "raw_hindi": corpus_bleu([row["english_from_raw"] or "" for row in predictions], references),
            "polished_hindi": corpus_bleu([row["english_from_polished"] or "" for row in predictions], references),
        },
        "by_category": {
            name: aggregate_transliteration(
                [row["transliteration"] for row in rows], [row["reference_devanagari"] for row in rows]
            ).__dict__
            for name, rows in by_category.items()
        },
        "human_review_instruction": "For each row, add adequacy_1_to_5, fluency_1_to_5, and reviewer_note to the predictions file.",
    }
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    (artifacts / "evaluation_predictions.json").write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifacts / "evaluation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
