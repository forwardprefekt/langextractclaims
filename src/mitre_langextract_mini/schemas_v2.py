from __future__ import annotations

from typing import Any, Final

CLAIM_STATES: Final[tuple[str, ...]] = (
    "fact",
    "corroborated",
    "hunch",
    "disputed",
    "disproven",
)

ASSOCIATION_STANCES: Final[tuple[str, ...]] = (
    "support",
    "contradict",
    "neutral",
)

STATE_PRIORITY: Final[dict[str, int]] = {
    "fact": 0,
    "corroborated": 1,
    "hunch": 2,
    "disputed": 3,
    "disproven": 4,
}


def normalize_claim_state(value: str) -> str:
    text = str(value or "").strip().lower()
    if text in CLAIM_STATES:
        return text
    return "hunch"


def normalize_association_stance(value: str) -> str:
    text = str(value or "").strip().lower()
    if text in ASSOCIATION_STANCES:
        return text
    return "neutral"


def summarize_claim_states(rows: list[dict[str, Any]]) -> dict[str, int]:
    summary = {state: 0 for state in CLAIM_STATES}
    for row in rows:
        state = normalize_claim_state(str(row.get("state") or ""))
        summary[state] += 1
    summary["total"] = len(rows)
    return summary
