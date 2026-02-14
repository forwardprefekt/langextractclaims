from __future__ import annotations

import pytest

from mitre_langextract_mini import evaluate

EVAL_DEPS_AVAILABLE = (
    evaluate.pytrec_eval is not None
    and evaluate.f1_score is not None
    and evaluate.confusion_matrix is not None
)


def _payload() -> dict[str, list[dict[str, object]]]:
    return {
        "claim_assessments": [
            {
                "claim_id": "claim_1",
                "claim_text": "Operators used PowerShell.",
                "state": "fact",
            },
            {
                "claim_id": "claim_2",
                "claim_text": "Operations were unaffected.",
                "state": "disproven",
            },
        ],
        "claim_associations": [
            {
                "claim_id": "claim_1",
                "association_id": "a1",
                "stance": "support",
                "association_score": 0.9,
                "start": 0,
                "end": 20,
                "quote": "PowerShell execution observed.",
            },
            {
                "claim_id": "claim_2",
                "association_id": "a2",
                "stance": "contradict",
                "association_score": 0.88,
                "start": 30,
                "end": 60,
                "quote": "Operations were disrupted.",
            },
        ],
    }


@pytest.mark.skipif(not EVAL_DEPS_AVAILABLE, reason="eval extras are not installed")
def test_evaluate_predictions_perfect_match():
    payload = _payload()
    result = evaluate.evaluate_predictions(payload, payload)

    assert result["claim_state"]["macro_f1"] == 1.0
    assert result["association_stance"]["macro_f1"] == 1.0
    assert result["retrieval"]["map"] == 1.0
    assert result["retrieval"]["recall_1"] == 1.0


@pytest.mark.skipif(not EVAL_DEPS_AVAILABLE, reason="eval extras are not installed")
def test_evaluate_predictions_missing_retrieval_degrades_recall():
    gold = _payload()
    pred = {
        "claim_assessments": gold["claim_assessments"],
        "claim_associations": [],
    }
    result = evaluate.evaluate_predictions(pred, gold)

    assert result["claim_state"]["macro_f1"] == 1.0
    assert result["retrieval"]["recall_1"] < 1.0
    assert result["retrieval"]["map"] < 1.0


@pytest.mark.skipif(not EVAL_DEPS_AVAILABLE, reason="eval extras are not installed")
def test_evaluate_predictions_stance_confusion_is_reflected():
    gold = _payload()
    pred = _payload()
    pred["claim_associations"][0]["stance"] = "contradict"

    result = evaluate.evaluate_predictions(pred, gold)

    assert result["association_stance"]["macro_f1"] < 1.0
    labels = result["association_stance"]["confusion_matrix"]["labels"]
    values = result["association_stance"]["confusion_matrix"]["values"]
    support_idx = labels.index("support")
    contradict_idx = labels.index("contradict")
    assert values[support_idx][contradict_idx] >= 1


@pytest.mark.skipif(not EVAL_DEPS_AVAILABLE, reason="eval extras are not installed")
def test_evaluate_predictions_quote_fallback_matching_without_offsets():
    gold = {
        "claim_assessments": [
            {
                "claim_id": "claim_1",
                "claim_text": "The campaign used phishing.",
                "state": "corroborated",
            }
        ],
        "claim_associations": [
            {
                "claim_id": "claim_1",
                "association_id": "g1",
                "stance": "support",
                "quote": "The campaign used phishing emails.",
            }
        ],
    }
    pred = {
        "claim_assessments": [
            {
                "claim_id": "claim_1",
                "claim_text": "The campaign used phishing.",
                "state": "corroborated",
            }
        ],
        "claim_associations": [
            {
                "claim_id": "claim_1",
                "association_id": "p1",
                "stance": "support",
                "association_score": 0.7,
                "quote": "The campaign used phishing emails.",
            }
        ],
    }

    result = evaluate.evaluate_predictions(pred, gold)
    assert result["association_stance"]["macro_f1"] == 1.0


def test_evaluate_dependencies_error_when_metric_libs_missing(monkeypatch):
    monkeypatch.setattr(evaluate, "pytrec_eval", None)
    with monkeypatch.context() as m:
        m.setattr(evaluate, "f1_score", None)
        m.setattr(evaluate, "confusion_matrix", None)
        try:
            evaluate.ensure_eval_dependencies()
            raise AssertionError("expected dependency error")
        except RuntimeError as exc:
            assert "pytrec_eval" in str(exc)
            assert "scikit-learn" in str(exc)
