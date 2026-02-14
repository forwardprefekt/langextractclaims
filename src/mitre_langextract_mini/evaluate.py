from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .schemas_v2 import ASSOCIATION_STANCES, CLAIM_STATES, normalize_association_stance, normalize_claim_state

try:
    import pytrec_eval  # type: ignore
except Exception:
    pytrec_eval = None

try:
    from sklearn.metrics import confusion_matrix, f1_score
except Exception:
    confusion_matrix = None
    f1_score = None


def ensure_eval_dependencies() -> None:
    missing: list[str] = []
    if pytrec_eval is None:
        missing.append("pytrec_eval")
    if f1_score is None or confusion_matrix is None:
        missing.append("scikit-learn")
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"evaluate requires optional dependencies: {joined}. "
            "Install package extras or dependencies in the active environment first."
        )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        obj = json.loads(text)
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _claim_key(row: dict[str, Any]) -> str:
    explicit = str(row.get("claim_id") or "").strip()
    if explicit:
        return explicit
    doc_id = str(row.get("document_id") or row.get("reference_id") or "")
    text = _normalize_text(str(row.get("claim_text") or row.get("text") or ""))
    mitre = str(row.get("mitre_id") or "")
    seed = f"{doc_id}|{text}|{mitre}"
    return "claim_" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]


def _association_id(row: dict[str, Any], *, fallback_prefix: str = "assoc") -> str:
    explicit = str(row.get("association_id") or row.get("citation_id") or "").strip()
    if explicit:
        return explicit
    start = row.get("start")
    end = row.get("end")
    quote = _normalize_text(str(row.get("quote") or ""))[:180]
    seed = f"{start}|{end}|{quote}"
    return f"{fallback_prefix}_" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]


def _load_eval_payload(path: Path) -> dict[str, list[dict[str, Any]]]:
    if path.is_dir():
        claims_path = path / "claims.jsonl"
        associations_path = path / "associations.jsonl"
        if not claims_path.exists():
            raise RuntimeError(f"Missing required file: {claims_path}")
        if not associations_path.exists():
            raise RuntimeError(f"Missing required file: {associations_path}")
        claim_rows = _load_jsonl(claims_path)
        assoc_rows = _load_jsonl(associations_path)
        return {
            "claim_assessments": claim_rows,
            "claim_associations": assoc_rows,
        }

    if path.suffix.lower() == ".jsonl":
        rows = _load_jsonl(path)
        if rows and ("state" in rows[0] or "claim_text" in rows[0]):
            return {
                "claim_assessments": rows,
                "claim_associations": [],
            }
        return {
            "claim_assessments": [],
            "claim_associations": rows,
        }

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unsupported payload format in {path}")

    claim_rows = payload.get("claim_assessments")
    assoc_rows = payload.get("claim_associations")
    if not isinstance(claim_rows, list):
        raise RuntimeError("Payload must include list field 'claim_assessments' (V2 schema).")
    if not isinstance(assoc_rows, list):
        raise RuntimeError("Payload must include list field 'claim_associations' (V2 schema).")

    return {
        "claim_assessments": [r for r in claim_rows if isinstance(r, dict)],
        "claim_associations": [r for r in assoc_rows if isinstance(r, dict)],
    }


def _span_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    a_start = a.get("start")
    a_end = a.get("end")
    b_start = b.get("start")
    b_end = b.get("end")
    if not all(isinstance(v, int) for v in (a_start, a_end, b_start, b_end)):
        return 0.0
    if not (a_start < a_end and b_start < b_end):
        return 0.0
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    if inter <= 0:
        return 0.0
    union = max(a_end, b_end) - min(a_start, b_start)
    if union <= 0:
        return 0.0
    return float(inter / union)


def _quote_similarity(a: dict[str, Any], b: dict[str, Any]) -> float:
    a_text = _normalize_text(str(a.get("quote") or ""))
    b_text = _normalize_text(str(b.get("quote") or ""))
    if not a_text or not b_text:
        return 0.0
    if a_text in b_text or b_text in a_text:
        return 1.0
    a_tokens = set(a_text.split())
    b_tokens = set(b_text.split())
    if not a_tokens or not b_tokens:
        return 0.0
    return float(len(a_tokens & b_tokens) / max(1, len(a_tokens | b_tokens)))


def _match_score(pred: dict[str, Any], gold: dict[str, Any]) -> float:
    iou = _span_iou(pred, gold)
    if iou > 0:
        return iou
    return _quote_similarity(pred, gold)


def _score_for_association(row: dict[str, Any]) -> float:
    raw = row.get("association_score")
    try:
        if raw is not None:
            out = float(raw)
            return max(0.0, min(out, 1.0))
    except Exception:
        pass
    stance = normalize_association_stance(str(row.get("stance") or ""))
    if stance == "support":
        return 0.8
    if stance == "contradict":
        return 0.75
    return 0.2


def _group_rows_by_claim(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = _claim_key(row)
        grouped.setdefault(key, []).append(dict(row))
    return grouped


def _build_claim_maps(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = _claim_key(row)
        item = dict(row)
        item["_claim_key"] = key
        item["state"] = normalize_claim_state(str(item.get("state") or ""))
        out[key] = item
    return out


def _match_associations(
    pred_rows: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    *,
    iou_threshold: float,
    quote_threshold: float,
) -> tuple[dict[int, int], set[int], set[int]]:
    scored: list[tuple[float, int, int]] = []
    for p_idx, pred in enumerate(pred_rows):
        for g_idx, gold in enumerate(gold_rows):
            score = _match_score(pred, gold)
            if _span_iou(pred, gold) > 0:
                if score < iou_threshold:
                    continue
            else:
                if score < quote_threshold:
                    continue
            scored.append((score, p_idx, g_idx))
    scored.sort(reverse=True)

    matched_pred: set[int] = set()
    matched_gold: set[int] = set()
    assignment: dict[int, int] = {}
    for _, p_idx, g_idx in scored:
        if p_idx in matched_pred or g_idx in matched_gold:
            continue
        matched_pred.add(p_idx)
        matched_gold.add(g_idx)
        assignment[p_idx] = g_idx

    unmatched_pred = set(range(len(pred_rows))) - matched_pred
    unmatched_gold = set(range(len(gold_rows))) - matched_gold
    return assignment, unmatched_pred, unmatched_gold


def _evaluate_claim_states(
    pred_claims: list[dict[str, Any]],
    gold_claims: list[dict[str, Any]],
) -> dict[str, Any]:
    pred_map = _build_claim_maps(pred_claims)
    gold_map = _build_claim_maps(gold_claims)

    y_true: list[str] = []
    y_pred: list[str] = []
    for key, gold in gold_map.items():
        y_true.append(normalize_claim_state(str(gold.get("state") or "")))
        pred_state = normalize_claim_state(str((pred_map.get(key) or {}).get("state") or "hunch"))
        y_pred.append(pred_state)

    macro = f1_score(y_true, y_pred, labels=list(CLAIM_STATES), average="macro", zero_division=0)
    per_class_raw = f1_score(y_true, y_pred, labels=list(CLAIM_STATES), average=None, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=list(CLAIM_STATES)).tolist()

    per_class = {
        label: float(score)
        for label, score in zip(CLAIM_STATES, per_class_raw)
    }
    return {
        "macro_f1": float(macro),
        "per_class_f1": per_class,
        "confusion_matrix": {
            "labels": list(CLAIM_STATES),
            "values": matrix,
        },
        "gold_count": len(gold_map),
        "pred_count": len(pred_map),
    }


def _evaluate_association_stances(
    pred_associations: list[dict[str, Any]],
    gold_associations: list[dict[str, Any]],
    *,
    iou_threshold: float,
    quote_threshold: float,
) -> dict[str, Any]:
    pred_by_claim = _group_rows_by_claim(pred_associations)
    gold_by_claim = _group_rows_by_claim(gold_associations)

    y_true: list[str] = []
    y_pred: list[str] = []

    claim_keys = sorted(set(gold_by_claim) | set(pred_by_claim))
    for claim_key in claim_keys:
        pred_rows = [dict(r) for r in pred_by_claim.get(claim_key, [])]
        gold_rows = [dict(r) for r in gold_by_claim.get(claim_key, [])]
        assignment, unmatched_pred, unmatched_gold = _match_associations(
            pred_rows,
            gold_rows,
            iou_threshold=iou_threshold,
            quote_threshold=quote_threshold,
        )

        for p_idx, g_idx in assignment.items():
            y_true.append(normalize_association_stance(str(gold_rows[g_idx].get("stance") or "")))
            y_pred.append(normalize_association_stance(str(pred_rows[p_idx].get("stance") or "")))

        for g_idx in unmatched_gold:
            y_true.append(normalize_association_stance(str(gold_rows[g_idx].get("stance") or "")))
            y_pred.append("neutral")

        for p_idx in unmatched_pred:
            y_true.append("neutral")
            y_pred.append(normalize_association_stance(str(pred_rows[p_idx].get("stance") or "")))

    if not y_true:
        y_true = ["neutral"]
        y_pred = ["neutral"]

    macro = f1_score(y_true, y_pred, labels=list(ASSOCIATION_STANCES), average="macro", zero_division=0)
    per_class_raw = f1_score(y_true, y_pred, labels=list(ASSOCIATION_STANCES), average=None, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=list(ASSOCIATION_STANCES)).tolist()

    per_class = {
        label: float(score)
        for label, score in zip(ASSOCIATION_STANCES, per_class_raw)
    }
    return {
        "macro_f1": float(macro),
        "per_class_f1": per_class,
        "confusion_matrix": {
            "labels": list(ASSOCIATION_STANCES),
            "values": matrix,
        },
        "comparisons": len(y_true),
    }


def _evaluate_retrieval(
    pred_associations: list[dict[str, Any]],
    gold_associations: list[dict[str, Any]],
    *,
    iou_threshold: float,
    quote_threshold: float,
) -> dict[str, Any]:
    pred_by_claim = _group_rows_by_claim(pred_associations)
    gold_by_claim = _group_rows_by_claim(gold_associations)
    claim_keys = sorted(set(gold_by_claim) | set(pred_by_claim))

    qrel: dict[str, dict[str, int]] = {}
    run: dict[str, dict[str, float]] = {}

    for claim_key in claim_keys:
        pred_rows = [dict(r) for r in pred_by_claim.get(claim_key, [])]
        gold_rows = [dict(r) for r in gold_by_claim.get(claim_key, [])]

        gold_ids: list[str] = []
        rel_by_gold_id: dict[str, int] = {}
        for g_idx, gold_row in enumerate(gold_rows):
            gid = _association_id(gold_row, fallback_prefix=f"gold_{claim_key}_{g_idx}")
            gold_ids.append(gid)
            stance = normalize_association_stance(str(gold_row.get("stance") or ""))
            rel_by_gold_id[gid] = 0 if stance == "neutral" else 1

        # Skip queries without relevant associations.
        if not any(v > 0 for v in rel_by_gold_id.values()):
            continue

        assignment, _, _ = _match_associations(
            pred_rows,
            gold_rows,
            iou_threshold=iou_threshold,
            quote_threshold=quote_threshold,
        )

        query_run: dict[str, float] = {}
        for p_idx, pred_row in enumerate(pred_rows):
            if p_idx in assignment:
                gid = gold_ids[assignment[p_idx]]
            else:
                gid = _association_id(pred_row, fallback_prefix=f"pred_{claim_key}_{p_idx}")
            query_run[gid] = max(query_run.get(gid, 0.0), _score_for_association(pred_row))

        if not query_run:
            query_run = {f"none_{claim_key}": 0.0}

        qrel[claim_key] = rel_by_gold_id
        run[claim_key] = query_run

    if not qrel:
        return {
            "map": 0.0,
            "recip_rank": 0.0,
            "ndcg_cut_5": 0.0,
            "recall_1": 0.0,
            "recall_3": 0.0,
            "recall_5": 0.0,
            "query_count": 0,
        }

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrel,
        {"map", "recip_rank", "ndcg_cut_5", "recall_1", "recall_3", "recall_5"},
    )
    per_query = evaluator.evaluate(run)

    def _avg(metric: str) -> float:
        values = [float(row.get(metric, 0.0)) for row in per_query.values()]
        return float(sum(values) / len(values)) if values else 0.0

    return {
        "map": _avg("map"),
        "recip_rank": _avg("recip_rank"),
        "ndcg_cut_5": _avg("ndcg_cut_5"),
        "recall_1": _avg("recall_1"),
        "recall_3": _avg("recall_3"),
        "recall_5": _avg("recall_5"),
        "query_count": len(per_query),
    }


def evaluate_predictions(
    pred_payload: dict[str, list[dict[str, Any]]],
    gold_payload: dict[str, list[dict[str, Any]]],
    *,
    span_iou_threshold: float = 0.5,
    quote_similarity_threshold: float = 0.8,
) -> dict[str, Any]:
    ensure_eval_dependencies()

    pred_claims = pred_payload.get("claim_assessments") or []
    pred_assocs = pred_payload.get("claim_associations") or []
    gold_claims = gold_payload.get("claim_assessments") or []
    gold_assocs = gold_payload.get("claim_associations") or []

    claim_state_metrics = _evaluate_claim_states(pred_claims, gold_claims)
    stance_metrics = _evaluate_association_stances(
        pred_assocs,
        gold_assocs,
        iou_threshold=span_iou_threshold,
        quote_threshold=quote_similarity_threshold,
    )
    retrieval_metrics = _evaluate_retrieval(
        pred_assocs,
        gold_assocs,
        iou_threshold=span_iou_threshold,
        quote_threshold=quote_similarity_threshold,
    )

    return {
        "meta": {
            "span_iou_threshold": float(span_iou_threshold),
            "quote_similarity_threshold": float(quote_similarity_threshold),
            "pred_claims": len(pred_claims),
            "gold_claims": len(gold_claims),
            "pred_associations": len(pred_assocs),
            "gold_associations": len(gold_assocs),
        },
        "claim_state": claim_state_metrics,
        "association_stance": stance_metrics,
        "retrieval": retrieval_metrics,
    }


def evaluate_prediction_files(
    *,
    pred_path: str | Path,
    gold_path: str | Path,
    span_iou_threshold: float = 0.5,
    quote_similarity_threshold: float = 0.8,
) -> dict[str, Any]:
    pred = _load_eval_payload(Path(pred_path))
    gold = _load_eval_payload(Path(gold_path))
    return evaluate_predictions(
        pred,
        gold,
        span_iou_threshold=span_iou_threshold,
        quote_similarity_threshold=quote_similarity_threshold,
    )


def to_markdown_report(result: dict[str, Any]) -> str:
    meta = result.get("meta") or {}
    claim = result.get("claim_state") or {}
    stance = result.get("association_stance") or {}
    retrieval = result.get("retrieval") or {}

    lines = [
        "# Claim Flow V2 Evaluation",
        "",
        "## Meta",
        f"- pred_claims: {meta.get('pred_claims', 0)}",
        f"- gold_claims: {meta.get('gold_claims', 0)}",
        f"- pred_associations: {meta.get('pred_associations', 0)}",
        f"- gold_associations: {meta.get('gold_associations', 0)}",
        f"- span_iou_threshold: {meta.get('span_iou_threshold', 0.0)}",
        f"- quote_similarity_threshold: {meta.get('quote_similarity_threshold', 0.0)}",
        "",
        "## Claim State",
        f"- macro_f1: {claim.get('macro_f1', 0.0):.6f}",
        "",
        "## Association Stance",
        f"- macro_f1: {stance.get('macro_f1', 0.0):.6f}",
        f"- comparisons: {stance.get('comparisons', 0)}",
        "",
        "## Retrieval",
        f"- map: {retrieval.get('map', 0.0):.6f}",
        f"- recip_rank: {retrieval.get('recip_rank', 0.0):.6f}",
        f"- ndcg_cut_5: {retrieval.get('ndcg_cut_5', 0.0):.6f}",
        f"- recall_1: {retrieval.get('recall_1', 0.0):.6f}",
        f"- recall_3: {retrieval.get('recall_3', 0.0):.6f}",
        f"- recall_5: {retrieval.get('recall_5', 0.0):.6f}",
        f"- query_count: {retrieval.get('query_count', 0)}",
    ]
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mitre-langextract-mini evaluate",
        description="Evaluate V2 claim assessments and associations against a gold payload.",
    )
    parser.add_argument("--pred", required=True, help="Prediction JSON/JSONL file or directory.")
    parser.add_argument("--gold", required=True, help="Gold JSON/JSONL file or directory.")
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    parser.add_argument("--span-iou-threshold", type=float, default=0.5)
    parser.add_argument("--quote-similarity-threshold", type=float, default=0.8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = evaluate_prediction_files(
        pred_path=args.pred,
        gold_path=args.gold,
        span_iou_threshold=float(args.span_iou_threshold),
        quote_similarity_threshold=float(args.quote_similarity_threshold),
    )
    if args.format == "markdown":
        print(to_markdown_report(result))
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
