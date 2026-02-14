from __future__ import annotations

import json
from pathlib import Path

from mitre_langextract_mini.schemas_v2 import ASSOCIATION_STANCES, CLAIM_STATES


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        obj = json.loads(text)
        assert isinstance(obj, dict)
        rows.append(obj)
    return rows


def test_eval_gold_fixture_shape_and_counts():
    root = Path("packages/mitre-langextract-mini/data/eval_gold/v1")
    claims = _load_jsonl(root / "claims.jsonl")
    associations = _load_jsonl(root / "associations.jsonl")
    links = _load_jsonl(root / "mitre_links.jsonl")

    assert len(claims) == 75
    assert len({str(row.get("document_id") or "") for row in claims}) == 10
    assert len(links) == 75
    assert len(associations) >= 75


def test_eval_gold_claims_schema():
    root = Path("packages/mitre-langextract-mini/data/eval_gold/v1")
    claims = _load_jsonl(root / "claims.jsonl")
    allowed_states = set(CLAIM_STATES)

    for row in claims:
        assert str(row.get("claim_id") or "")
        assert str(row.get("document_id") or "")
        assert str(row.get("claim_text") or "")
        assert str(row.get("mitre_id") or "")
        assert str(row.get("state") or "") in allowed_states


def test_eval_gold_associations_schema_and_fk_integrity():
    root = Path("packages/mitre-langextract-mini/data/eval_gold/v1")
    claims = _load_jsonl(root / "claims.jsonl")
    associations = _load_jsonl(root / "associations.jsonl")
    claim_ids = {str(row.get("claim_id") or "") for row in claims}
    allowed_stances = set(ASSOCIATION_STANCES)

    for row in associations:
        claim_id = str(row.get("claim_id") or "")
        assert claim_id in claim_ids
        assert str(row.get("association_id") or "")
        assert str(row.get("stance") or "") in allowed_stances
        assert isinstance(row.get("start"), int)
        assert isinstance(row.get("end"), int)
        assert int(row.get("end") or 0) > int(row.get("start") or 0)
        assert str(row.get("quote") or "")
