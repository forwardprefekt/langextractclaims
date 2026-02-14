from __future__ import annotations

import json
from pathlib import Path

import pytest

from mitre_langextract_mini import evaluate, workflow


@pytest.mark.parametrize(
    "path",
    [
        Path("notebooks/mitre_langextract_playground.ipynb"),
        Path("packages/mitre-langextract-mini/README.md"),
    ],
)
def test_consumer_assets_exist(path: Path):
    assert path.exists(), f"missing consumer asset: {path}"


def test_notebook_smoke_uses_v2_keys():
    text = Path("notebooks/mitre_langextract_playground.ipynb").read_text(encoding="utf-8")
    assert "claim_assessments" in text
    assert "claim_associations" in text
    assert "claims_table" not in text
    assert "claim_references_table" not in text
    assert "validating" not in text
    assert "contradicting" not in text


def test_readme_examples_use_subcommands_and_v2_tables():
    text = Path("packages/mitre-langextract-mini/README.md").read_text(encoding="utf-8")
    assert "mitre-langextract-mini extract" in text
    assert "mitre-langextract-mini evaluate" in text
    assert "claim_assessments" in text
    assert "claim_associations" in text
    assert "claims_table" not in text
    assert "claim_references_table" not in text


def test_readme_evaluate_snippet_executes_with_v2_payload(monkeypatch, capsys, tmp_path):
    if evaluate.pytrec_eval is None or evaluate.f1_score is None or evaluate.confusion_matrix is None:
        pytest.skip("eval extras are not installed")

    pred_payload = {
        "claim_assessments": [
            {
                "claim_id": "claim_1",
                "claim_text": "Operators used PowerShell.",
                "state": "fact",
            }
        ],
        "claim_associations": [
            {
                "claim_id": "claim_1",
                "association_id": "assoc_1",
                "stance": "support",
                "association_score": 0.9,
                "start": 0,
                "end": 10,
                "quote": "PowerShell",
            }
        ],
    }

    pred_path = tmp_path / "pred.json"
    pred_path.write_text(json.dumps(pred_payload), encoding="utf-8")

    gold_dir = Path("packages/mitre-langextract-mini/data/eval_gold/v1")
    code = workflow.main(["evaluate", "--pred", str(pred_path), "--gold", str(gold_dir), "--format", "json"])
    captured = capsys.readouterr()

    assert code == 0
    parsed = json.loads(captured.out)
    assert "claim_state" in parsed
    assert "association_stance" in parsed
    assert "retrieval" in parsed
