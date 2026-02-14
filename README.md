# mitre-langextract-mini

MITRE ATT&CK reference extraction plus Claim Flow V2 evaluation.

## What It Does

1. Fetches one ATT&CK reference document (random or explicit URL).
2. Extracts ATT&CK technique mentions and falsifiable claims.
3. Builds grounded claim citations with offsets and association stance.
4. Produces V2 outputs aligned to GraphFS claim taxonomy.

## Claim Flow V2 Contract

Canonical claim states:
- `fact`
- `corroborated`
- `hunch`
- `disputed`
- `disproven`

Canonical association stances:
- `support`
- `contradict`
- `neutral`

Top-level V2 keys:
- `technique_mentions`
- `claims`
- `claim_assessments`
- `claim_associations`
- `assessment_summary`

Claim object uses `assessment` (not `validation`):
- `state`
- `confidence`
- `support_score`
- `contradict_score`
- `support_count`
- `contradict_count`
- `cite_count`
- `reason`

## Install

```bash
uv tool install --from ./packages/mitre-langextract-mini mitre-langextract-mini --force
```

Notebook extras:

```bash
uv sync --directory packages/mitre-langextract-mini --extra notebook
```

Evaluation extras:

```bash
uv sync --directory packages/mitre-langextract-mini --extra eval
```

## CLI

The CLI uses explicit subcommands.

### Extract

```bash
mitre-langextract-mini extract --reference-url "https://therecord.media/dutch-telecom-giant-announces-data-breach"
```

```bash
mitre-langextract-mini extract --no-langextract --no-nli-stances
```

Inspect V2 tables:

```bash
mitre-langextract-mini extract --reference-url "https://example.com/report" | jq '.claim_assessments'
```

```bash
mitre-langextract-mini extract --reference-url "https://example.com/report" | jq '.claim_associations'
```

### Evaluate

```bash
mitre-langextract-mini evaluate \
  --pred packages/mitre-langextract-mini/data/eval_gold/v1 \
  --gold packages/mitre-langextract-mini/data/eval_gold/v1 \
  --format json
```

```bash
mitre-langextract-mini evaluate \
  --pred packages/mitre-langextract-mini/data/eval_gold/v1 \
  --gold packages/mitre-langextract-mini/data/eval_gold/v1 \
  --format markdown
```

The evaluator expects V2 schema (`claim_assessments` + `claim_associations`).

## Metrics

`evaluate` reports:
- claim-state classification: macro/per-class F1 + confusion matrix
- association-stance classification: macro/per-class F1 + confusion matrix
- retrieval metrics (`pytrec_eval`): `map`, `recip_rank`, `ndcg_cut_5`, `recall_1`, `recall_3`, `recall_5`

## Runtime Notes

- Provider default: `openai`
- Model default: `gpt-5-mini` (or `OPENAI_MODEL`)
- Requires `OPENAI_API_KEY` unless running `--no-langextract`
- NLI stance check is on by default (`--no-nli-stances` to disable)
- Wayback-first fetch is default (`--direct-first` to invert)
- `pytrec_eval` may fail to build on Python `3.13`; prefer Python `3.11`/`3.12` for `--extra eval`.

## Tests (Full Output)

```bash
PYTHONPATH=packages/mitre-langextract-mini/src .venv/bin/python -m pytest -vv -s packages/mitre-langextract-mini/tests
```

## Curated Gold

In-repo gold set:
- `packages/mitre-langextract-mini/data/eval_gold/v1/claims.jsonl`
- `packages/mitre-langextract-mini/data/eval_gold/v1/associations.jsonl`
- `packages/mitre-langextract-mini/data/eval_gold/v1/mitre_links.jsonl`
