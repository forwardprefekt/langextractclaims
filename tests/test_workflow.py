from __future__ import annotations

import io
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
from mitre_langextract_mini import workflow


def test_run_random_reference_langextract_orchestrates(monkeypatch, tmp_path):
    bundle = {
        "objects": [
            {
                "type": "attack-pattern",
                "name": "Command and Scripting Interpreter: PowerShell",
                "external_references": [
                    {"source_name": "mitre-attack", "external_id": "T1059.001"},
                    {"source_name": "example", "url": "https://example.com/report"},
                ],
            }
        ]
    }
    downloaded = tmp_path / "reference.txt"
    downloaded.write_text("PowerShell usage in incident report.", encoding="utf-8")

    monkeypatch.setattr(workflow, "download_attack_bundle", lambda **_: bundle)
    monkeypatch.setattr(
        workflow,
        "download_random_reference",
        lambda *_args, **_kwargs: {
            "path": str(downloaded),
            "content_type": "text/plain",
            "resolved_url": "https://example.com/report",
            "download_source": "direct",
        },
    )
    monkeypatch.setattr(
        workflow,
        "run_langextract",
        lambda document, **_: [
            {
                "extraction_class": "mitre_technique",
                "extraction_text": "PowerShell",
                "attributes": {"mitre_technique_id": "T1059.001"},
            }
        ],
    )
    monkeypatch.setattr(workflow, "_maybe_import_langextract", lambda: object())

    result = workflow.run_random_reference_langextract(
        model_id="google/gemini-1.5-flash",
        cache_path=Path(tmp_path / "cache.json"),
        output_dir=tmp_path,
        max_chars=999,
        timeout=7,
    )

    assert result["reference"]["path"].endswith(".txt")
    assert "PowerShell" in result["document"]
    assert result["technique_mentions"][0]["mitre_id"] == "T1059.001"
    assert result["technique_mentions"][0]["claim"]["assessment"]["state"] in {
        "fact",
        "corroborated",
        "hunch",
        "disputed",
        "disproven",
    }
    assert result["claims"][0]["mitre_id"] == "T1059.001"
    assert result["assessment_summary"]["overall"]["total"] >= 1
    assert result["status"]["techniques"] == 1
    assert result["status"]["results_source"] == "langextract"
    assert result["status"]["langextract_error"] is None


def test_run_random_reference_langextract_uses_reference_url_override(monkeypatch, tmp_path):
    bundle = {
        "objects": [
            {
                "type": "attack-pattern",
                "name": "Command and Scripting Interpreter: PowerShell",
                "external_references": [
                    {"source_name": "mitre-attack", "external_id": "T1059.001"},
                ],
            }
        ]
    }
    downloaded = tmp_path / "manual.txt"
    downloaded.write_text("PowerShell activity observed.", encoding="utf-8")

    seen: dict[str, str] = {}

    monkeypatch.setattr(workflow, "download_attack_bundle", lambda **_: bundle)
    monkeypatch.setattr(
        workflow,
        "download_reference_from_url",
        lambda url, **_kwargs: (
            seen.update({"url": url})
            or {
                "meta": {
                    "mitre_id": "",
                    "technique_name": "",
                    "source_name": "manual",
                    "url": url,
                },
                "path": str(downloaded),
                "content_type": "text/plain",
                "resolved_url": url,
                "download_source": "direct",
            }
        ),
    )
    monkeypatch.setattr(
        workflow,
        "download_random_reference",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("random path should not be used")),
    )
    monkeypatch.setattr(
        workflow,
        "run_langextract",
        lambda document, **_: [
            {
                "extraction_class": "mitre_technique",
                "extraction_text": "PowerShell",
                "attributes": {"mitre_technique_id": "T1059.001"},
            }
        ],
    )

    url = "https://example.com/custom-report"
    result = workflow.run_random_reference_langextract(
        cache_path=Path(tmp_path / "cache.json"),
        output_dir=tmp_path,
        timeout=7,
        reference_url=url,
    )

    assert seen["url"] == url
    assert result["reference"]["resolved_url"] == url
    assert result["technique_mentions"][0]["mitre_id"] == "T1059.001"
    assert result["status"]["reference_url_override"] == url


def test_to_cli_payload_builds_preview_by_default():
    payload = workflow.to_cli_payload(
        {
            "status": {"techniques": 42},
            "reference": {"path": "x"},
            "document": "abcdef",
            "technique_mentions": [],
        },
        include_document=False,
        preview_chars=3,
    )

    assert "document" not in payload
    assert payload["document_chars"] == 6
    assert payload["document_preview"] == "abc"


def test_run_langextract_openai_applies_recommended_kwargs(monkeypatch):
    seen: dict[str, object] = {}

    def _fake_extract(**kwargs):
        seen.update(kwargs)
        return {
            "extractions": [
                {
                    "extraction_class": "mitre_technique",
                    "extraction_text": "PowerShell",
                    "attributes": {"mitre_technique_id": "T1059.001"},
                }
            ]
        }

    fake_lx = SimpleNamespace(extract=_fake_extract)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    rows = workflow.run_langextract(
        "doc text",
        provider="openai",
        model_id=None,
        lx_module=fake_lx,
    )

    assert rows and rows[0]["attributes"]["mitre_technique_id"] == "T1059.001"
    assert seen["api_key"] == "test-openai-key"
    assert seen["fence_output"] is False
    assert seen["use_schema_constraints"] is True
    assert seen["model_id"] == "gpt-5-mini"
    assert seen["language_model_params"]["reasoning_effort"] == "high"
    assert seen["show_progress"] is False


def test_run_langextract_openai_requires_api_key(monkeypatch):
    fake_lx = SimpleNamespace(extract=lambda **kwargs: kwargs)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        workflow.run_langextract(
            "doc text",
            provider="openai",
            model_id="gpt-5-mini",
            lx_module=fake_lx,
        )


def test_run_langextract_openai_allows_reasoning_override(monkeypatch):
    seen: dict[str, object] = {}

    def _fake_extract(**kwargs):
        seen.update(kwargs)
        return {"extractions": []}

    fake_lx = SimpleNamespace(extract=_fake_extract)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    workflow.run_langextract(
        "doc text",
        provider="openai",
        model_id="gpt-5-mini",
        openai_reasoning_effort="minimal",
        lx_module=fake_lx,
    )

    assert seen["language_model_params"]["reasoning_effort"] == "minimal"


def test_run_langextract_unwraps_wrapper_signature(monkeypatch):
    seen: dict[str, object] = {}

    def _strict_extract(
        text_or_documents,
        prompt_description=None,
        examples=None,
        model_id=None,
        api_key=None,
        language_model_params=None,
        fence_output=None,
        use_schema_constraints=True,
    ):
        seen["text_or_documents"] = text_or_documents
        seen["prompt_description"] = prompt_description
        seen["model_id"] = model_id
        seen["api_key"] = api_key
        seen["language_model_params"] = language_model_params
        seen["fence_output"] = fence_output
        seen["use_schema_constraints"] = use_schema_constraints
        return {"extractions": []}

    def _wrapper_extract(*args, **kwargs):
        return _strict_extract(*args, **kwargs)

    # Mimic langextract.__init__.extract wrapper pattern.
    _wrapper_extract.__module__ = workflow.__name__
    monkeypatch.setattr(workflow, "extract_func", _strict_extract, raising=False)

    fake_lx = SimpleNamespace(extract=_wrapper_extract)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    rows = workflow.run_langextract(
        "doc text",
        provider="openai",
        model_id="gpt-5-mini",
        lx_module=fake_lx,
    )

    assert rows == []
    assert seen["text_or_documents"] == "doc text"
    assert seen["prompt_description"] == workflow.DEFAULT_PROMPT
    assert seen["api_key"] == "test-openai-key"


def test_run_langextract_openai_falls_back_when_reasoning_not_supported(monkeypatch):
    seen: list[dict[str, object]] = []

    def _fake_extract(**kwargs):
        seen.append(dict(kwargs))
        lm_params = kwargs.get("language_model_params") or {}
        if lm_params.get("reasoning_effort"):
            raise RuntimeError(
                "OpenAI API error: Completions.create() got an unexpected keyword argument 'reasoning'"
            )
        return {"extractions": []}

    fake_lx = SimpleNamespace(extract=_fake_extract)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    rows = workflow.run_langextract(
        "doc text",
        provider="openai",
        model_id="gpt-5-mini",
        lx_module=fake_lx,
    )

    assert rows == []
    assert len(seen) >= 2
    assert seen[0].get("language_model_params", {}).get("reasoning_effort") == "high"
    assert "language_model_params" not in seen[-1] or not seen[-1]["language_model_params"]


def test_run_langextract_openai_falls_back_to_schema_constraints_on_parse_errors(monkeypatch):
    seen: list[dict[str, object]] = []

    def _fake_extract(**kwargs):
        seen.append(dict(kwargs))
        if kwargs.get("use_schema_constraints") is True:
            raise RuntimeError("ResolverParsingError: Content must contain an 'extractions' key.")
        return {"extractions": []}

    fake_lx = SimpleNamespace(extract=_fake_extract)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    rows = workflow.run_langextract(
        "doc text",
        provider="openai",
        model_id="gpt-5-mini",
        lx_module=fake_lx,
    )

    assert rows == []
    assert any(call.get("use_schema_constraints") is True for call in seen)
    assert any(call.get("use_schema_constraints") is False for call in seen)


def test_run_langextract_show_progress_flag(monkeypatch):
    seen: dict[str, object] = {}

    def _fake_extract(**kwargs):
        seen.update(kwargs)
        return {"extractions": []}

    fake_lx = SimpleNamespace(extract=_fake_extract)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    workflow.run_langextract(
        "doc text",
        provider="openai",
        model_id="gpt-5-mini",
        show_progress=True,
        lx_module=fake_lx,
    )

    assert seen["show_progress"] is True


def test_run_random_reference_langextract_falls_back_to_id_matching(monkeypatch, tmp_path):
    bundle = {
        "objects": [
            {
                "type": "attack-pattern",
                "name": "Command and Scripting Interpreter: PowerShell",
                "external_references": [
                    {"source_name": "mitre-attack", "external_id": "T1059.001"},
                    {"source_name": "example", "url": "https://example.com/report"},
                ],
                "kill_chain_phases": [
                    {"kill_chain_name": "mitre-attack", "phase_name": "execution"}
                ],
            }
        ]
    }
    downloaded = tmp_path / "reference.txt"
    downloaded.write_text("Observed behavior includes T1059.001 in logs.", encoding="utf-8")

    monkeypatch.setattr(workflow, "download_attack_bundle", lambda **_: bundle)
    monkeypatch.setattr(
        workflow,
        "download_random_reference",
        lambda *_args, **_kwargs: {
            "path": str(downloaded),
            "content_type": "text/plain",
            "resolved_url": "https://example.com/report",
            "download_source": "direct",
        },
    )
    monkeypatch.setattr(
        workflow,
        "run_langextract",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("ResolverParsingError: Content must contain an 'extractions' key.")
        ),
    )

    result = workflow.run_random_reference_langextract(
        cache_path=Path(tmp_path / "cache.json"),
        output_dir=tmp_path,
        timeout=7,
    )

    assert result["status"]["results_source"] == "fallback_id_match"
    assert "ResolverParsingError" in str(result["status"]["langextract_error"])
    assert result["technique_mentions"][0]["source"] == "fallback_id_match"
    assert result["technique_mentions"][0]["mitre_id"] == "T1059.001"
    assert result["claims"][0]["mitre_id"] == "T1059.001"


def test_claim_citation_linkage_is_coherent():
    doc = "Observed T1059.001 PowerShell execution in process logs."
    row = {
        "mitre_id": "T1059.001",
        "technique_name": "PowerShell",
        "tactic": "execution",
        "extraction_text": "PowerShell",
        "confidence": "high",
        "source": "langextract",
    }

    enriched, claims = workflow._enrich_rows_with_claims(doc, [row])
    assert len(enriched) == 1
    assert len(claims) == 1
    citation_ids = {c["citation_id"] for c in enriched[0]["citations"]}
    assert citation_ids
    assert set(claims[0]["citation_ids"]).issubset(citation_ids)
    first_citation = enriched[0]["citations"][0]
    assert first_citation.get("relation_to_mitre")
    assert first_citation.get("validation_explanation")
    assert (enriched[0].get("assessment") or {}).get("reason")


def test_unmapped_cyber_claim_is_self_referential():
    doc = "The company's operations have not been impacted by the attack."
    row = {
        "mitre_id": "",
        "technique_name": "",
        "tactic": "",
        "extraction_text": "The company's operations have not been impacted by the attack.",
        "claim_text": "The company's operations have not been impacted by the attack.",
        "confidence": "medium",
        "source": "langextract_claim_first_pass",
        "claim_scope": "cybersecurity",
        "validation_method": "mitre_data",
    }

    enriched, claims = workflow._enrich_rows_with_claims(doc, [row])
    assert enriched and claims
    assessment = enriched[0]["assessment"]
    assert assessment["state"] in {"hunch", "corroborated", "fact", "disputed", "disproven"}
    assert isinstance(assessment["confidence"], float)
    first_citation = enriched[0]["citations"][0]
    assert "mapped MITRE technique" not in str(first_citation.get("validation_explanation") or "")
    assert claims[0]["validation_method"] == "self_referential"


def test_citation_stance_detects_negation_contradiction():
    claim_text = "The attack impacted operations."
    citation = {
        "citation_id": "cite_1",
        "method": "contextual",
        "quote": "The attack did not impact operations.",
    }
    assert workflow._citation_stance_for_claim(claim_text, citation) == "contradict"


def test_run_random_reference_langextract_uses_metadata_claim_when_empty(monkeypatch, tmp_path):
    bundle = {
        "objects": [
            {
                "type": "attack-pattern",
                "name": "PowerShell",
                "external_references": [
                    {"source_name": "mitre-attack", "external_id": "T1059.001"},
                    {"source_name": "example", "url": "https://example.com/report"},
                ],
            }
        ]
    }
    downloaded = tmp_path / "reference.txt"
    downloaded.write_text("No explicit ATT&CK ids in this text.", encoding="utf-8")

    monkeypatch.setattr(workflow, "download_attack_bundle", lambda **_: bundle)
    monkeypatch.setattr(
        workflow,
        "download_random_reference",
        lambda *_args, **_kwargs: {
            "meta": {
                "mitre_id": "T1059.001",
                "technique_name": "PowerShell",
                "source_name": "example",
                "url": "https://example.com/report",
            },
            "path": str(downloaded),
            "content_type": "text/plain",
            "resolved_url": "https://example.com/report",
            "download_source": "direct",
        },
    )
    monkeypatch.setattr(workflow, "run_langextract", lambda *args, **kwargs: [])

    result = workflow.run_random_reference_langextract(
        cache_path=Path(tmp_path / "cache.json"),
        output_dir=tmp_path,
        timeout=7,
    )

    assert result["status"]["results_source"] == "reference_metadata"
    assert result["claims"][0]["mitre_id"] == "T1059.001"
    assert result["technique_mentions"][0]["source"] == "reference_metadata"


def test_run_random_reference_langextract_drops_ungrounded_langextract_rows(monkeypatch, tmp_path):
    bundle = {
        "objects": [
            {
                "type": "attack-pattern",
                "name": "Temporary Elevated Cloud Access",
                "external_references": [
                    {"source_name": "mitre-attack", "external_id": "T1548.005"},
                    {"source_name": "AWS_PassRole", "url": "https://example.com/passrole"},
                ],
                "kill_chain_phases": [
                    {"kill_chain_name": "mitre-attack", "phase_name": "privilege-escalation"}
                ],
            }
        ]
    }
    downloaded = tmp_path / "reference.txt"
    downloaded.write_text(
        "Granting iam:PassRole lets a user pass a role to an AWS service.",
        encoding="utf-8",
    )

    monkeypatch.setattr(workflow, "download_attack_bundle", lambda **_: bundle)
    monkeypatch.setattr(
        workflow,
        "download_random_reference",
        lambda *_args, **_kwargs: {
            "meta": {
                "mitre_id": "T1548.005",
                "technique_name": "Temporary Elevated Cloud Access",
                "source_name": "AWS_PassRole",
                "url": "https://example.com/passrole",
            },
            "path": str(downloaded),
            "content_type": "text/plain",
            "resolved_url": "https://example.com/passrole",
            "download_source": "direct",
        },
    )
    # Ungrounded model mapping: technique id/name not present in text.
    monkeypatch.setattr(
        workflow,
        "run_langextract",
        lambda *args, **kwargs: [
            {
                "extraction_class": "mitre_technique",
                "extraction_text": "IAM PassRole",
                "attributes": {"mitre_technique_id": "T1078", "confidence": "high"},
            }
        ],
    )

    result = workflow.run_random_reference_langextract(
        cache_path=Path(tmp_path / "cache.json"),
        output_dir=tmp_path,
        timeout=7,
    )

    assert result["status"]["results_source"] == "reference_metadata"
    assert result["claims"][0]["mitre_id"] == "T1548.005"
    assert result["technique_mentions"][0]["citations"]
    first_quote = result["technique_mentions"][0]["citations"][0]["quote"]
    assert "PassRole".lower() in first_quote.lower()


def test_normalize_extracted_rows_coerces_dict_extraction_text():
    rows = [
        {
            "extraction_class": "mitre_technique",
            "extraction_text": {"text": "PowerShell"},
            "attributes": {"mitre_technique_id": "T1059.001"},
        }
    ]
    lookup = {"T1059.001": {"name": "PowerShell", "tactic": "execution"}}

    normalized = workflow._normalize_extracted_rows(rows, lookup)
    assert normalized[0]["extraction_text"] == "PowerShell"
    assert normalized[0]["mitre_id"] == "T1059.001"


def test_falsifiable_claims_have_validation_method_and_linkage_offsets(monkeypatch, tmp_path):
    bundle = {
        "objects": [
            {
                "type": "attack-pattern",
                "name": "Command and Scripting Interpreter: PowerShell",
                "external_references": [
                    {"source_name": "mitre-attack", "external_id": "T1059.001"},
                    {"source_name": "example", "url": "https://example.com/report"},
                ],
            }
        ]
    }
    downloaded = tmp_path / "reference.txt"
    downloaded.write_text(
        "The mayor announced a citywide curfew after midnight. "
        "Operators used T1059.001 to execute PowerShell scripts.",
        encoding="utf-8",
    )

    monkeypatch.setattr(workflow, "download_attack_bundle", lambda **_: bundle)
    monkeypatch.setattr(
        workflow,
        "download_random_reference",
        lambda *_args, **_kwargs: {
            "path": str(downloaded),
            "content_type": "text/plain",
            "resolved_url": "https://example.com/report",
            "download_source": "direct",
        },
    )
    monkeypatch.setattr(workflow, "run_langextract", lambda *args, **kwargs: [])

    result = workflow.run_random_reference_langextract(
        cache_path=Path(tmp_path / "cache.json"),
        output_dir=tmp_path,
        timeout=7,
    )

    falsifiable_claims = result.get("falsifiable_claims") or []
    assert falsifiable_claims
    assert any(c.get("validation_method") == "self_referential" for c in falsifiable_claims)
    assert any(c.get("validation_method") == "mitre_data" for c in falsifiable_claims)

    linkages = result.get("mitre_linkages") or []
    assert linkages
    first_citation = (linkages[0].get("citations") or [])[0]
    assert isinstance(first_citation.get("start"), int)
    assert isinstance(first_citation.get("end"), int)
    assert first_citation["end"] > first_citation["start"]
    assert first_citation.get("relation_to_mitre")
    assert first_citation.get("validation_explanation")
    assert (linkages[0].get("assessment") or {}).get("reason")


def test_result_contains_consolidated_claim_tables(monkeypatch, tmp_path):
    bundle = {
        "objects": [
            {
                "type": "attack-pattern",
                "name": "Command and Scripting Interpreter: PowerShell",
                "external_references": [
                    {"source_name": "mitre-attack", "external_id": "T1059.001"},
                    {"source_name": "example", "url": "https://example.com/report"},
                ],
            }
        ]
    }
    downloaded = tmp_path / "reference.txt"
    downloaded.write_text(
        "Analysts reported that operators used T1059.001 for PowerShell execution. "
        "The company said operations were not impacted.",
        encoding="utf-8",
    )

    monkeypatch.setattr(workflow, "download_attack_bundle", lambda **_: bundle)
    monkeypatch.setattr(
        workflow,
        "download_random_reference",
        lambda *_args, **_kwargs: {
            "path": str(downloaded),
            "content_type": "text/plain",
            "resolved_url": "https://example.com/report",
            "download_source": "direct",
        },
    )
    monkeypatch.setattr(workflow, "run_langextract", lambda *args, **kwargs: [])

    result = workflow.run_random_reference_langextract(
        cache_path=Path(tmp_path / "cache.json"),
        output_dir=tmp_path,
        timeout=7,
    )

    claim_assessments = result.get("claim_assessments") or []
    claim_associations = result.get("claim_associations") or []
    assert claim_assessments
    assert claim_associations
    first_claim_row = claim_assessments[0]
    assert "support_references" in first_claim_row
    assert "contradict_references" in first_claim_row
    assert isinstance(first_claim_row["support_count"], int)
    assert isinstance(first_claim_row["contradict_count"], int)

    first_ref_row = claim_associations[0]
    assert first_ref_row.get("claim_id")
    assert first_ref_row.get("citation_id")
    assert first_ref_row.get("stance") in {"support", "contradict", "neutral"}
    assert isinstance(first_ref_row.get("association_score"), float)
    assert "reference_url" in first_ref_row
    assert "quote" in first_ref_row


def test_consolidation_tables_apply_nli_stance_override():
    claim = {
        "claim_id": "claim_1",
        "text": "The attack impacted operations.",
        "citation_ids": ["cite_1"],
        "assessment": {"state": "corroborated", "confidence": 0.5, "reason": "x"},
        "validation_method": "self_referential",
        "claim_scope": "general",
        "falsifiable": True,
        "source": "unit",
        "mitre_id": "",
        "technique_name": "",
        "tactic": "",
    }
    claim_rows = [
        {
            "claim": claim,
            "citations": [
                {
                    "citation_id": "cite_1",
                    "method": "extraction_text",
                    "quote": "The attack impacted operations.",
                    "context": "The attack impacted operations.",
                }
            ],
        }
    ]
    nli_override = {
        ("claim_1", "cite_1"): {
            "label": "neutral",
            "stance": "neutral",
            "confidence": 0.91,
            "reason": "insufficient specificity",
        }
    }

    claim_assessments, claim_associations = workflow._build_claim_consolidation_tables(
        [claim],
        claim_rows,
        reference_url="https://example.com",
        nli_stance_labels=nli_override,
    )

    assert claim_associations and claim_associations[0]["stance"] == "neutral"
    assert claim_associations[0]["stance_source"] == "nli"
    assert claim_assessments[0]["neutral_reference_count"] == 1
    assert claim_assessments[0]["support_reference_count"] == 0


def test_nli_stance_labels_for_pairs_skips_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    labels, status, usage = workflow._nli_stance_labels_for_pairs(
        [
            {
                "claim_id": "claim_1",
                "citation_id": "cite_1",
                "claim_text": "A",
                "citation_quote": "A",
                "citation_context": "A",
            }
        ],
        openai_api_key=None,
    )
    assert labels == {}
    assert "missing OPENAI_API_KEY" in str(status)
    assert usage["total_tokens"] == 0


def test_run_random_reference_langextract_exposes_token_usage_sections(monkeypatch, tmp_path):
    bundle = {
        "objects": [
            {
                "type": "attack-pattern",
                "name": "Command and Scripting Interpreter: PowerShell",
                "external_references": [
                    {"source_name": "mitre-attack", "external_id": "T1059.001"},
                ],
            }
        ]
    }
    downloaded = tmp_path / "reference.txt"
    downloaded.write_text("Observed T1059.001 activity in logs.", encoding="utf-8")

    monkeypatch.setattr(workflow, "download_attack_bundle", lambda **_: bundle)
    monkeypatch.setattr(
        workflow,
        "download_random_reference",
        lambda *_args, **_kwargs: {
            "path": str(downloaded),
            "content_type": "text/plain",
            "resolved_url": "https://example.com/report",
            "download_source": "direct",
            "fetch_attempts": [
                {
                    "source": "direct",
                    "url": "https://example.com/report",
                    "status": "ok",
                    "token_usage": {
                        "input_tokens": 123,
                        "output_tokens": 17,
                        "cached_tokens": 50,
                        "total_tokens": 140,
                    },
                }
            ],
        },
    )
    monkeypatch.setattr(workflow, "run_langextract", lambda *args, **kwargs: [])

    result = workflow.run_random_reference_langextract(
        cache_path=Path(tmp_path / "cache.json"),
        output_dir=tmp_path,
        timeout=7,
        nli_stance_check=False,
    )

    token_usage = result.get("token_usage") or {}
    overall = token_usage.get("overall") or {}
    sections = token_usage.get("sections") or {}
    assert overall.get("input_tokens") == 123
    assert overall.get("output_tokens") == 17
    assert overall.get("cached_tokens") == 50
    assert overall.get("total_tokens") == 140
    assert (sections.get("paywall_check") or {}).get("total_tokens") == 140
    assert (sections.get("nli_stance") or {}).get("total_tokens") == 0
    assert (sections.get("langextract") or {}).get("available") is False
    assert (result.get("status") or {}).get("token_usage_overall", {}).get("total_tokens") == 140


def test_download_reference_from_url_wayback_first(monkeypatch, tmp_path):
    class _Resp:
        def __init__(self, body: str):
            self.headers = {"content-type": "text/html"}
            self.content = body.encode("utf-8")
            self.text = body

    direct_url = "https://example.com/article"
    wayback_url = "https://web.archive.org/web/20260101010101/https://example.com/article"

    monkeypatch.setattr(workflow, "_wayback_candidates", lambda *_args, **_kwargs: [wayback_url])

    def _fake_fetch(url, **_kwargs):
        if url == direct_url:
            return _Resp("PAYWALL CONTENT")
        if url == wayback_url:
            return _Resp("Archived open content")
        raise RuntimeError("unexpected url")

    monkeypatch.setattr(workflow, "_fetch_or_raise", _fake_fetch)
    monkeypatch.setattr(
        workflow,
        "_is_response_paywalled",
        lambda _target, response, **_kwargs: ("paywall" in response.text.lower(), "detected"),
    )

    out = workflow.download_reference_from_url(
        direct_url,
        output_dir=tmp_path,
        prefer_wayback=True,
        timeout=7,
    )

    assert out["download_source"] == "wayback"
    assert out["resolved_url"] == wayback_url
    assert wayback_url in out["wayback_candidates_available"]
    assert wayback_url in out["wayback_candidates_considered"]
    attempts = out.get("fetch_attempts") or []
    assert len(attempts) >= 1
    assert attempts[-1]["status"] == "ok"
    assert attempts[-1]["source"] == "wayback"


def test_main_missing_openai_key_returns_clean_error(monkeypatch, capsys):
    monkeypatch.setattr(
        workflow,
        "run_random_reference_langextract",
        lambda **_: (_ for _ in ()).throw(
            RuntimeError("OpenAI mode requires OPENAI_API_KEY (or pass --openai-api-key).")
        ),
    )
    code = workflow.main(["extract"])
    captured = capsys.readouterr()
    assert code == 2
    assert "ERROR:" in captured.err
    assert "OPENAI_API_KEY" in captured.err


def test_configure_logging_defaults_to_info():
    workflow.configure_logging()
    assert workflow.LOGGER.level == logging.INFO


def test_configure_logging_handles_closed_stderr_stream_without_noise(capsys):
    workflow.configure_logging()
    stream_handlers = [h for h in workflow.LOGGER.handlers if isinstance(h, logging.StreamHandler)]
    assert stream_handlers

    stale_stream = io.StringIO()
    stream_handlers[0].stream = stale_stream
    stale_stream.close()

    workflow.LOGGER.info("closed stream safety check")
    captured = capsys.readouterr()
    assert "Logging error" not in captured.err


def test_parser_defaults_to_wayback_first_and_allows_direct_first():
    parser = workflow._build_parser()
    args_default = parser.parse_args(["extract"])
    assert args_default.wayback_first is True
    assert args_default.llm_paywall_check is True
    assert args_default.nli_stance_check is True

    args_direct = parser.parse_args(["extract", "--direct-first"])
    assert args_direct.wayback_first is False
    assert args_direct.llm_paywall_check is True
    assert args_direct.nli_stance_check is True

    args_no_llm = parser.parse_args(["extract", "--no-llm-paywall-check"])
    assert args_no_llm.llm_paywall_check is False

    args_no_nli = parser.parse_args(["extract", "--no-nli-stances"])
    assert args_no_nli.nli_stance_check is False


def test_parser_supports_evaluate_subcommand():
    parser = workflow._build_parser()
    args = parser.parse_args(["evaluate", "--pred", "pred.json", "--gold", "gold.json"])
    assert args.command == "evaluate"
    assert args.pred == "pred.json"
    assert args.gold == "gold.json"
    assert args.format == "json"


def test_main_evaluate_subcommand_prints_json(monkeypatch, capsys):
    monkeypatch.setattr(
        workflow.eval_v2,
        "evaluate_prediction_files",
        lambda **_: {
            "meta": {"pred_claims": 1, "gold_claims": 1},
            "claim_state": {"macro_f1": 1.0},
            "association_stance": {"macro_f1": 1.0},
            "retrieval": {"map": 1.0},
        },
    )
    code = workflow.main(["evaluate", "--pred", "pred.json", "--gold", "gold.json"])
    captured = capsys.readouterr()
    assert code == 0
    assert "\"claim_state\"" in captured.out


def test_download_reference_from_url_failure_reason_timeout(monkeypatch, tmp_path):
    monkeypatch.setattr(workflow, "_wayback_candidates", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        workflow,
        "_fetch_or_raise",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(requests.Timeout("slow network")),
    )

    with pytest.raises(workflow.ReferenceFetchError) as excinfo:
        workflow.download_reference_from_url(
            "https://example.com/resource",
            output_dir=tmp_path,
            prefer_wayback=False,
            timeout=1,
        )

    assert excinfo.value.failure_reason == "timeout"
    assert "failure_reason=timeout" in str(excinfo.value)


def test_download_reference_from_url_failure_reason_paywall(monkeypatch, tmp_path):
    class _Resp:
        def __init__(self, body: str):
            self.headers = {"content-type": "text/html"}
            self.content = body.encode("utf-8")
            self.text = body

    monkeypatch.setattr(workflow, "_wayback_candidates", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(workflow, "_fetch_or_raise", lambda *_args, **_kwargs: _Resp("paywall text"))
    monkeypatch.setattr(workflow, "_is_response_paywalled", lambda *_args, **_kwargs: (True, "paywall marker"))

    with pytest.raises(workflow.ReferenceFetchError) as excinfo:
        workflow.download_reference_from_url(
            "https://example.com/resource",
            output_dir=tmp_path,
            prefer_wayback=False,
            timeout=7,
        )

    assert excinfo.value.failure_reason == "paywall"
    assert "failure_reason=paywall" in str(excinfo.value)


def test_classify_fetch_exception_404():
    class _Resp:
        status_code = 404

    exc = requests.HTTPError("404", response=_Resp())
    assert workflow._classify_fetch_exception(exc) == "404"
