from __future__ import annotations

import argparse
import concurrent.futures
from contextlib import nullcontext
import hashlib
import inspect
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from . import evaluate as eval_v2
from .schemas_v2 import (
    ASSOCIATION_STANCES,
    CLAIM_STATES,
    STATE_PRIORITY,
    normalize_association_stance,
    normalize_claim_state,
    summarize_claim_states,
)


ATTACK_STIX_URL = (
    "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/"
    "enterprise-attack.json"
)
MITRE_CACHE = Path.home() / ".cache" / "mitre_enterprise_attack.json"
DEFAULT_USER_AGENT = "mitre-langextract-mini/1.0"
DEFAULT_OPENAI_MODEL = "gpt-5-mini"
DEFAULT_OPENAI_REASONING_EFFORT = "high"
DEFAULT_LOG_LEVEL = "info"
LOG_LEVEL_CHOICES = ("debug", "info", "warning", "error", "critical")
LOGGER = logging.getLogger("mitre_langextract_mini")
DEFAULT_WAYBACKPY_TIMEOUT_SECONDS = 12
DEFAULT_PROMPT = """
You extract ATT&CK technique mentions from text.

Return only high-confidence technique evidence as structured entries with:
- extraction_class = mitre_technique
- extraction_text = exact source span
- attributes.mitre_technique_id = ATT&CK ID when present in text (T#### or T####.###)
- attributes.tactic = ATT&CK tactic if explicit
- attributes.confidence = high|medium|low

Do not infer technique IDs unless text supports it.
""".strip()
DEFAULT_EXAMPLE_TEXTS = [
    (
        "Attackers used PowerShell and C2 over web sessions to stage persistence.",
        [
            {
                "extraction_class": "mitre_technique",
                "extraction_text": "PowerShell",
                "attributes": {
                    "mitre_technique_id": "T1059.001",
                    "tactic": "command-and-control",
                    "confidence": "high",
                },
            }
        ],
    )
]
CLAIM_PASS_PROMPT = """
You extract falsifiable claims from text.

Return each claim as:
- extraction_class = falsifiable_claim
- extraction_text = exact claim sentence span copied from source text
- attributes.claim_scope = cybersecurity|general
- attributes.validation_method = mitre_data for cybersecurity claims, self_referential otherwise
- attributes.mitre_technique_id = ATT&CK ID if explicitly present in source text or clearly named
- attributes.confidence = high|medium|low

Rules:
- Claims must be verifiable from the source document text alone.
- Do not invent entities or events not present in the text.
- Keep extraction_text verbatim from source text.
""".strip()
CLAIM_PASS_EXAMPLE_TEXTS = [
    (
        "The report says attackers used phishing emails to steal credentials. "
        "The campaign also used T1566.001 against employees.",
        [
            {
                "extraction_class": "falsifiable_claim",
                "extraction_text": "The report says attackers used phishing emails to steal credentials.",
                "attributes": {
                    "claim_scope": "cybersecurity",
                    "validation_method": "mitre_data",
                    "mitre_technique_id": "T1566",
                    "confidence": "high",
                },
            },
            {
                "extraction_class": "falsifiable_claim",
                "extraction_text": "The campaign also used T1566.001 against employees.",
                "attributes": {
                    "claim_scope": "cybersecurity",
                    "validation_method": "mitre_data",
                    "mitre_technique_id": "T1566.001",
                    "confidence": "high",
                },
            },
        ],
    )
]
MITRE_ID_PATTERN = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")
_CLAIM_VERB_PATTERN = re.compile(
    r"\b("
    r"is|are|was|were|be|been|being|"
    r"has|have|had|"
    r"do|does|did|"
    r"can|could|will|would|should|may|might|must|"
    r"uses|used|use|"
    r"reports?|reported|states?|stated|says?|said|"
    r"announce(?:s|d)|"
    r"targets?|targeted|launch(?:es|ed)?|"
    r"carries|carried|perform(?:s|ed)|"
    r"contains?|contained|"
    r"detect(?:s|ed)|blocks?|blocked|"
    r"access(?:es|ed)?|exfiltrat(?:es|ed)"
    r")\b",
    flags=re.IGNORECASE,
)
_CYBER_KEYWORDS = (
    "attack",
    "attacker",
    "breach",
    "c2",
    "cloudtrail",
    "credential",
    "cyber",
    "defense",
    "endpoint",
    "exfiltration",
    "exploit",
    "hack",
    "hacker",
    "iam",
    "incident",
    "intrusion",
    "malware",
    "mitre",
    "network",
    "payload",
    "persistence",
    "phishing",
    "privilege",
    "ransomware",
    "security",
    "tactic",
    "technique",
    "threat",
    "vulnerability",
)
_NEGATION_PATTERN = re.compile(
    r"\b(no|not|never|none|without|deny|denied|denies|isn't|aren't|wasn't|weren't|can't|cannot|won't|didn't|doesn't|don't)\b",
    flags=re.IGNORECASE,
)
_NLI_LABEL_CANONICAL = {"entailment", "contradiction", "neutral"}
PAYWALL_DOMAIN_HINTS = {
    "ft.com",
    "haaretz.com",
    "jpost.com",
    "nytimes.com",
    "thetimes.co.uk",
    "wsj.com",
}
PAYWALL_TEXT_HINTS = (
    "subscribe to continue",
    "subscription required",
    "already a subscriber",
    "log in to continue reading",
    "you have reached your free",
    "to continue reading this article",
    "exclusive to subscribers",
    "sign in to continue",
    "this content is for subscribers",
    "purchase a subscription",
    "metered paywall",
)


class ReferenceFetchError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        failure_reason: str,
        attempts: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.failure_reason = str(failure_reason or "unknown")
        self.attempts = attempts or []


def _coerce_log_level(level_name: str | None) -> int:
    name = str(level_name or DEFAULT_LOG_LEVEL).strip().lower()
    mapping = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    return mapping.get(name, logging.INFO)


class _ResilientStderrHandler(logging.StreamHandler):
    """Stream to the current stderr and tolerate capture teardown."""

    def emit(self, record: logging.LogRecord) -> None:
        # pytest capture swaps/closes stderr between tests. Rebind on each emit
        # so module-level loggers don't retain stale stream objects.
        self.stream = sys.stderr
        try:
            super().emit(record)
        except ValueError:
            # Ignore late writes to closed capture streams.
            return


def configure_logging(level_name: str = DEFAULT_LOG_LEVEL) -> None:
    level = _coerce_log_level(level_name)
    for handler in list(LOGGER.handlers):
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, _ResilientStderrHandler):
            LOGGER.removeHandler(handler)
    if not any(isinstance(handler, _ResilientStderrHandler) for handler in LOGGER.handlers):
        handler = _ResilientStderrHandler(stream=sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        LOGGER.addHandler(handler)
    LOGGER.setLevel(level)
    LOGGER.propagate = False


def _extract_mitre_id(obj: dict[str, Any]) -> str | None:
    for ext in obj.get("external_references", []):
        if (
            ext.get("source_name") == "mitre-attack"
            and str(ext.get("external_id", "")).startswith("T")
        ):
            return str(ext.get("external_id"))
    return None


def _extract_tactic(obj: dict[str, Any]) -> str:
    for phase in obj.get("kill_chain_phases", []):
        if isinstance(phase, dict) and phase.get("kill_chain_name") == "mitre-attack":
            return str(phase.get("phase_name") or "")
    return ""


def _build_technique_lookup(bundle: dict[str, Any]) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for obj in bundle.get("objects", []):
        if obj.get("type") != "attack-pattern" or obj.get("revoked"):
            continue
        mitre_id = _extract_mitre_id(obj)
        if not mitre_id:
            continue
        lookup[mitre_id] = {
            "name": str(obj.get("name") or ""),
            "tactic": _extract_tactic(obj),
        }
    return lookup


def _claim_sentence_spans(document: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    text = str(document or "")
    if not text.strip():
        return spans

    # Sentence-ish splitting: punctuation-terminated chunks or paragraph breaks.
    for match in re.finditer(r"\S[\s\S]*?(?:[.!?](?=\s|$)|\n{2,}|$)", text):
        raw = text[match.start() : match.end()]
        if not raw:
            continue
        left_trim = len(raw) - len(raw.lstrip())
        right_trim = len(raw) - len(raw.rstrip())
        start = match.start() + left_trim
        end = match.end() - right_trim
        if end <= start:
            continue
        sentence = text[start:end]
        if sentence.strip():
            spans.append((start, end, sentence))
    return spans


def _is_falsifiable_sentence(sentence: str) -> bool:
    text = _compact_whitespace(sentence)
    if len(text) < 25:
        return False
    if text.endswith("?"):
        return False
    if text.lower().startswith(("http://", "https://")):
        return False
    word_count = len(re.findall(r"\b\w+\b", text))
    if word_count < 5:
        return False
    if text.lower().startswith(("copyright", "all rights reserved")):
        return False
    if _CLAIM_VERB_PATTERN.search(text):
        return True
    return bool(re.search(r"\b\d+\b", text))


def _is_cybersecurity_sentence(sentence: str) -> bool:
    text = str(sentence or "")
    if MITRE_ID_PATTERN.search(text):
        return True
    lower = text.lower()
    return any(keyword in lower for keyword in _CYBER_KEYWORDS)


def _build_technique_name_index(
    technique_lookup: dict[str, dict[str, str]],
) -> list[tuple[str, str, str, str]]:
    index: list[tuple[str, str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for mitre_id, record in technique_lookup.items():
        name = str(record.get("name") or "").strip()
        tactic = str(record.get("tactic") or "")
        if not name:
            continue
        variants = {name}
        if ":" in name:
            tail = name.split(":")[-1].strip()
            if tail:
                variants.add(tail)
        for variant in variants:
            variant_norm = variant.lower()
            key = (mitre_id, variant_norm)
            if key in seen:
                continue
            seen.add(key)
            index.append((variant_norm, mitre_id, name, tactic))
    index.sort(key=lambda item: len(item[0]), reverse=True)
    return index


def _map_sentence_to_mitre(
    sentence: str,
    technique_lookup: dict[str, dict[str, str]],
    technique_name_index: list[tuple[str, str, str, str]],
) -> tuple[str, str, str, str]:
    ids = MITRE_ID_PATTERN.findall(sentence or "")
    for mitre_id in ids:
        record = technique_lookup.get(mitre_id, {})
        return (
            mitre_id,
            str(record.get("name") or ""),
            str(record.get("tactic") or ""),
            "explicit_id",
        )

    lower = str(sentence or "").lower()
    for variant_norm, mitre_id, canonical_name, tactic in technique_name_index:
        if len(variant_norm) < 4:
            continue
        if variant_norm in lower:
            return mitre_id, canonical_name, tactic, "technique_name_match"
    return "", "", "", ""


def _extract_falsifiable_claim_rows(
    document: str,
    technique_lookup: dict[str, dict[str, str]],
    *,
    max_claims: int = 400,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    technique_name_index = _build_technique_name_index(technique_lookup)

    for start, end, sentence in _claim_sentence_spans(document):
        text = _compact_whitespace(sentence)
        if not _is_falsifiable_sentence(text):
            continue

        mitre_id, technique_name, tactic, match_method = _map_sentence_to_mitre(
            text,
            technique_lookup,
            technique_name_index,
        )
        is_cyber = _is_cybersecurity_sentence(text) or bool(mitre_id)
        rows.append(
            {
                "mitre_id": mitre_id,
                "technique_name": technique_name,
                "tactic": tactic,
                "extraction_text": text,
                "claim_text": text,
                "confidence": "high" if len(text) >= 50 else "medium",
                "source": "falsifiable_claim_extraction",
                "claim_scope": "cybersecurity" if is_cyber else "general",
                "validation_method": "mitre_data" if (is_cyber and mitre_id) else "self_referential",
                "mitre_match_method": match_method or "",
                "span_start": start,
                "span_end": end,
                "raw": {
                    "sentence_start": start,
                    "sentence_end": end,
                },
            }
        )
        if len(rows) >= max_claims:
            break
    return rows


def _fallback_extract_from_document(
    document: str,
    technique_lookup: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ids_in_text = sorted(set(MITRE_ID_PATTERN.findall(document or "")))
    for mitre_id in ids_in_text:
        rec = technique_lookup.get(mitre_id, {})
        rows.append(
            {
                "extraction_class": "mitre_technique",
                "extraction_text": mitre_id,
                "attributes": {
                    "mitre_technique_id": mitre_id,
                    "technique_name": rec.get("name", ""),
                    "tactic": rec.get("tactic", ""),
                    "confidence": "low",
                    "validation_method": "deterministic_id_match",
                },
                "source": "fallback_id_match",
            }
        )
    return rows


def _reference_anchor_text(meta: dict[str, Any], technique_name: str, *, document: str = "") -> str:
    source_name = str(meta.get("source_name") or "").strip()
    url = str(meta.get("url") or "").strip()
    document_lower = str(document or "").lower()

    token_candidates: list[str] = []
    phrase_candidates: list[str] = []
    if source_name:
        source_phrase = re.sub(r"[_-]+", " ", source_name).strip()
        if source_phrase:
            phrase_candidates.append(source_phrase)
            words = [part for part in re.split(r"\s+", source_phrase) if part]
            if len(words) >= 2:
                phrase_candidates.append(" ".join(words[-2:]))
            if len(words) >= 3:
                phrase_candidates.append(" ".join(words[-3:]))
        token_candidates.extend(re.split(r"[^A-Za-z0-9]+", source_name))
    if url:
        parsed = urlparse(url)
        stem = Path(parsed.path).stem
        if stem:
            phrase_candidates.append(re.sub(r"[_-]+", " ", stem).strip())
        token_candidates.extend(re.split(r"[^A-Za-z0-9]+", stem))

    for candidate in phrase_candidates:
        candidate = str(candidate or "").strip()
        if len(candidate) < 5:
            continue
        if document_lower and candidate.lower() in document_lower:
            return candidate

    noise = {
        "attack",
        "article",
        "aws",
        "blog",
        "cti",
        "doc",
        "docs",
        "documentation",
        "example",
        "guide",
        "mitre",
        "paper",
        "reference",
        "report",
        "security",
        "technique",
        "whitepaper",
    }
    cleaned = [
        token
        for token in token_candidates
        if token and len(token) >= 5 and token.lower() not in noise
    ]
    if document_lower:
        in_document = [token for token in cleaned if token.lower() in document_lower]
        if in_document:
            return max(in_document, key=len)
    if cleaned:
        # Prefer longer tokens; ties keep earlier appearance.
        return max(cleaned, key=len)

    if source_name:
        source_name = re.sub(r"[_-]+", " ", source_name).strip()
        if source_name:
            return source_name
    if technique_name:
        return technique_name
    return str(meta.get("mitre_id") or "")


def _reference_metadata_row(
    reference_download: dict[str, Any],
    technique_lookup: dict[str, dict[str, str]],
    *,
    document: str = "",
) -> dict[str, Any] | None:
    meta = reference_download.get("meta") if isinstance(reference_download.get("meta"), dict) else {}
    if not isinstance(meta, dict):
        return None

    meta_id = str(meta.get("mitre_id") or "").strip()
    if not meta_id:
        return None

    lookup_row = technique_lookup.get(meta_id, {})
    technique_name = str(lookup_row.get("name") or meta.get("technique_name") or "")
    tactic = str(lookup_row.get("tactic") or "")
    anchor = _reference_anchor_text(meta, technique_name, document=document)

    return {
        "mitre_id": meta_id,
        "technique_name": technique_name,
        "tactic": tactic,
        "extraction_text": anchor or meta_id,
        "confidence": "medium",
        "source": "reference_metadata",
        "raw": {"reference_meta": meta},
    }


def _normalize_extracted_rows(
    rows: list[dict[str, Any]],
    technique_lookup: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows or []:
        attrs = row.get("attributes") if isinstance(row, dict) else {}
        if not isinstance(attrs, dict):
            attrs = {}
        raw_extraction_text = row.get("extraction_text", "") if isinstance(row, dict) else ""
        if isinstance(raw_extraction_text, (str, int, float)):
            extraction_text = str(raw_extraction_text)
        elif isinstance(raw_extraction_text, dict):
            extraction_text = str(
                raw_extraction_text.get("text")
                or raw_extraction_text.get("value")
                or raw_extraction_text.get("name")
                or ""
            )
        elif isinstance(raw_extraction_text, list):
            extraction_text = " ".join(
                str(part)
                for part in raw_extraction_text
                if isinstance(part, (str, int, float))
            )
        else:
            extraction_text = ""
        mitre_id = str(attrs.get("mitre_technique_id", "") or "")
        if not mitre_id:
            ids = MITRE_ID_PATTERN.findall(extraction_text)
            if ids:
                mitre_id = ids[0]
        lookup_row = technique_lookup.get(mitre_id, {})
        technique_name = str(
            attrs.get("technique_name")
            or lookup_row.get("name")
            or row.get("technique_name")
            or ""
        )
        tactic = str(
            attrs.get("tactic")
            or lookup_row.get("tactic")
            or row.get("tactic")
            or ""
        )
        confidence = str(attrs.get("confidence") or row.get("confidence") or "medium").lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = "medium"
        normalized.append(
            {
                "mitre_id": mitre_id or "",
                "technique_name": technique_name,
                "tactic": tactic,
                "extraction_text": extraction_text,
                "confidence": confidence,
                "source": str(row.get("source") or "langextract"),
                "raw": row,
            }
        )
    return normalized


def _find_spans(
    document: str,
    phrase: str,
    *,
    whole_word: bool = False,
    max_hits: int = 4,
) -> list[tuple[int, int]]:
    phrase = str(phrase or "").strip()
    if not phrase:
        return []
    pattern = re.escape(phrase)
    if whole_word:
        pattern = rf"\b{pattern}\b"
    spans: list[tuple[int, int]] = []
    for match in re.finditer(pattern, document, flags=re.IGNORECASE):
        spans.append((match.start(), match.end()))
        if len(spans) >= max_hits:
            break
    return spans


def _context_for_span(document: str, start: int, end: int, *, max_chars: int = 260) -> str:
    left = max(
        document.rfind(".", 0, start),
        document.rfind("!", 0, start),
        document.rfind("?", 0, start),
        document.rfind("\n", 0, start),
    )
    right_candidates = [
        idx
        for idx in [
            document.find(".", end),
            document.find("!", end),
            document.find("?", end),
            document.find("\n", end),
        ]
        if idx != -1
    ]
    right = min(right_candidates) + 1 if right_candidates else len(document)
    left = 0 if left < 0 else left + 1
    snippet = document[left:right].strip()
    if not snippet:
        left = max(0, start - 120)
        right = min(len(document), end + 120)
        snippet = document[left:right].strip()
    return _compact_whitespace(snippet)[:max_chars]


def _build_citations_for_row(
    document: str,
    row: dict[str, Any],
    *,
    max_citations: int = 4,
) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()

    span_start = row.get("span_start")
    span_end = row.get("span_end")
    if isinstance(span_start, int) and isinstance(span_end, int):
        if 0 <= span_start < span_end <= len(document):
            seen.add((span_start, span_end))
            citations.append(
                {
                    "citation_id": "cite_1",
                    "method": "extraction_span",
                    "start": span_start,
                    "end": span_end,
                    "quote": document[span_start:span_end],
                    "context": _context_for_span(document, span_start, span_end),
                    "snippet": _context_for_span(document, span_start, span_end),
                }
            )

    specs: list[tuple[str, str, bool]] = []
    if row.get("extraction_text"):
        specs.append((str(row["extraction_text"]), "extraction_text", False))
    if row.get("mitre_id"):
        specs.append((str(row["mitre_id"]), "mitre_id", True))
    technique_name = str(row.get("technique_name", "") or "")
    if technique_name:
        whole_word = (" " not in technique_name) and len(technique_name) <= 14
        specs.append((technique_name, "technique_name", whole_word))

    for phrase, method, whole_word in specs:
        for start, end in _find_spans(document, phrase, whole_word=whole_word, max_hits=max_citations):
            if (start, end) in seen:
                continue
            seen.add((start, end))
            citations.append(
                {
                    "citation_id": f"cite_{len(citations) + 1}",
                    "method": method,
                    "start": start,
                    "end": end,
                    "quote": document[start:end],
                    "context": _context_for_span(document, start, end),
                    "snippet": _context_for_span(document, start, end),
                }
            )
            if len(citations) >= max_citations:
                return citations
    return citations


def _technique_label(row: dict[str, Any]) -> str:
    mitre_id = str(row.get("mitre_id") or "").strip()
    technique_name = str(row.get("technique_name") or "").strip()
    if mitre_id and technique_name:
        return f"{technique_name} ({mitre_id})"
    if mitre_id:
        return mitre_id
    if technique_name:
        return technique_name
    return "the mapped MITRE technique"


def _has_mitre_mapping(row: dict[str, Any]) -> bool:
    return bool(str(row.get("mitre_id") or "").strip() or str(row.get("technique_name") or "").strip())


def _resolve_validation_method(row: dict[str, Any]) -> str:
    requested = str(row.get("validation_method") or "").strip().lower()
    claim_scope = str(row.get("claim_scope") or "").strip().lower()
    has_mapping = _has_mitre_mapping(row)
    if requested == "mitre_data" and has_mapping:
        return "mitre_data"
    if requested == "self_referential":
        return "self_referential"
    if claim_scope == "cybersecurity" and has_mapping:
        return "mitre_data"
    return "self_referential"


def _citation_relation_and_explanation(
    row: dict[str, Any],
    citation: dict[str, Any],
) -> tuple[str, str]:
    method = str(citation.get("method") or "")
    quote = str(citation.get("quote") or "")
    quote_lower = quote.lower()
    mitre_id = str(row.get("mitre_id") or "").strip()
    technique_name = str(row.get("technique_name") or "").strip()
    technique_label = _technique_label(row)
    validation_method = _resolve_validation_method(row)

    has_explicit_id = bool(mitre_id and mitre_id.lower() in quote_lower)
    has_technique_name = bool(
        technique_name
        and _texts_overlap(quote, technique_name, threshold=0.65)
    )

    if method == "mitre_id" or has_explicit_id:
        return (
            "explicit_mitre_id_match",
            f"Citation explicitly contains ATT&CK ID {mitre_id}, directly grounding mapping to {technique_label}.",
        )
    if method == "technique_name" or has_technique_name:
        return (
            "technique_name_match",
            f"Citation includes the technique name for {technique_label}, grounding the MITRE linkage in source text.",
        )
    if validation_method == "mitre_data":
        if method in {"extraction_span", "extraction_text"}:
            return (
                "claim_span_support",
                f"Citation is the extracted claim span used to support mapping to {technique_label}.",
            )
        return (
            "contextual_support",
            f"Citation provides source context used to support mapping to {technique_label}.",
        )
    if validation_method == "self_referential":
        return (
            "self_referential_source_support",
            "Citation grounds the claim directly in source text; validation is self-referential.",
        )
    return (
        "source_support",
        "Citation provides grounded support from the source document text.",
    )


def _validation_explanation(row: dict[str, Any], citations: list[dict[str, Any]]) -> str:
    validation_method = _resolve_validation_method(row)
    if not citations:
        if validation_method == "mitre_data":
            return "No grounded citation span was found to justify the MITRE mapping."
        return "No grounded citation span was found for this claim."

    priority = {
        "explicit_mitre_id_match": 0,
        "technique_name_match": 1,
        "claim_span_support": 2,
        "contextual_support": 3,
        "self_referential_source_support": 4,
        "source_support": 5,
    }
    ranked = sorted(
        citations,
        key=lambda c: priority.get(str(c.get("relation_to_mitre") or ""), 99),
    )
    best = ranked[0] if ranked else {}
    explanation = str(best.get("validation_explanation") or "").strip()
    if explanation:
        return explanation

    relation, fallback = _citation_relation_and_explanation(row, best)
    if relation:
        return fallback
    return "Claim validation is based on grounded citation evidence in the source document."


def _texts_overlap(a: str, b: str, *, threshold: float = 0.6) -> bool:
    a_norm = re.sub(r"[^a-z0-9]+", " ", str(a or "").lower()).strip()
    b_norm = re.sub(r"[^a-z0-9]+", " ", str(b or "").lower()).strip()
    if not a_norm or not b_norm:
        return False
    if a_norm in b_norm or b_norm in a_norm:
        return True
    a_tokens = {tok for tok in a_norm.split() if tok}
    b_tokens = {tok for tok in b_norm.split() if tok}
    if not a_tokens or not b_tokens:
        return False
    overlap = len(a_tokens & b_tokens) / max(1, min(len(a_tokens), len(b_tokens)))
    return overlap >= threshold


def _association_score_for_citation(citation: dict[str, Any], stance: str) -> float:
    method = str(citation.get("method") or "").strip().lower()
    relation = str(citation.get("relation_to_mitre") or "").strip().lower()
    if stance == "neutral":
        return 0.2
    if method in {"extraction_span", "extraction_text"}:
        return 0.9
    if relation in {"explicit_mitre_id_match", "technique_name_match"}:
        return 0.95
    if relation in {"claim_span_support", "contextual_support"}:
        return 0.8
    if stance == "contradict":
        return 0.75
    return 0.7


def _assess_claim(
    row: dict[str, Any],
    citations: list[dict[str, Any]],
) -> dict[str, Any]:
    claim_text = str(row.get("claim_text") or row.get("extraction_text") or "")
    support_scores: list[float] = []
    contradict_scores: list[float] = []

    for citation in citations:
        if not isinstance(citation, dict):
            continue
        stance = normalize_association_stance(_citation_stance_for_claim(claim_text, citation))
        score = _association_score_for_citation(citation, stance)
        if stance == "support":
            support_scores.append(score)
        elif stance == "contradict":
            contradict_scores.append(score)

    support_score = float(sum(support_scores))
    contradict_score = float(sum(contradict_scores))
    support_count = len(support_scores)
    contradict_count = len(contradict_scores)
    cite_count = len(citations)
    support_avg = support_score / support_count if support_count else 0.0
    contradict_avg = contradict_score / contradict_count if contradict_count else 0.0

    state = "hunch"
    reason = "no_evidence"
    if support_count == 0 and contradict_count == 0:
        state = "hunch"
        reason = "no_evidence"
    elif support_count > 0 and contradict_count == 0:
        state = "fact" if support_score >= 1.0 else "corroborated"
        reason = "support_only"
    elif support_count == 0 and contradict_count > 0:
        state = "disproven"
        reason = "contradiction_only"
    else:
        if contradict_score >= (support_score + 0.5) and contradict_score > 0:
            state = "disproven"
            reason = "stronger_contradiction"
        else:
            state = "disputed"
            reason = "conflicting_evidence"

    if state in {"fact", "corroborated"}:
        confidence = max(support_avg, support_score / max(support_count, 1))
    elif state == "disproven":
        confidence = max(contradict_avg, contradict_score / max(contradict_count, 1))
    elif state == "disputed":
        confidence = max(support_avg, contradict_avg)
    else:
        confidence = 0.2

    return {
        "state": normalize_claim_state(state),
        "confidence": round(float(max(0.0, min(confidence, 1.0))), 3),
        "support_score": round(support_score, 3),
        "contradict_score": round(contradict_score, 3),
        "support_count": int(support_count),
        "contradict_count": int(contradict_count),
        "cite_count": int(cite_count),
        "reason": str(reason),
    }


def _build_claim(row: dict[str, Any], assessment: dict[str, Any], citations: list[dict[str, Any]]) -> dict[str, Any]:
    mitre_id = str(row.get("mitre_id", "") or "")
    technique_name = str(row.get("technique_name", "") or "")
    tactic = str(row.get("tactic", "") or "")
    explicit_claim_text = str(row.get("claim_text") or "").strip()
    if explicit_claim_text:
        text = explicit_claim_text
    elif mitre_id and technique_name:
        text = f"The document references ATT&CK technique {technique_name} ({mitre_id})."
    elif mitre_id:
        text = f"The document references ATT&CK technique {mitre_id}."
    elif technique_name:
        text = f"The document references ATT&CK technique behavior: {technique_name}."
    else:
        text = "The document contains a potential ATT&CK-relevant behavior."
    if tactic and not explicit_claim_text:
        text += f" Tactic: {tactic}."

    claim_key = f"{mitre_id}|{technique_name}|{row.get('extraction_text','')}|{row.get('source','')}"
    claim_id = "claim_" + hashlib.sha1(claim_key.encode("utf-8")).hexdigest()[:12]
    validation_method = _resolve_validation_method(row)
    return {
        "claim_id": claim_id,
        "text": text,
        "mitre_id": mitre_id,
        "technique_name": technique_name,
        "tactic": tactic,
        "claim_scope": str(row.get("claim_scope") or ("cybersecurity" if mitre_id else "general")),
        "falsifiable": True,
        "validation_method": validation_method,
        "assessment": assessment,
        "citation_ids": [c["citation_id"] for c in citations],
        "source": row.get("source", ""),
    }


def _citation_stance_for_claim(claim_text: str, citation: dict[str, Any]) -> str:
    quote = str(citation.get("quote") or "")
    method = str(citation.get("method") or "").strip().lower()
    relation = str(citation.get("relation_to_mitre") or "").strip().lower()
    if not quote.strip():
        return "neutral"

    if method in {"extraction_span", "extraction_text"}:
        return "support"

    overlap = _texts_overlap(claim_text, quote, threshold=0.5)
    claim_negated = bool(_NEGATION_PATTERN.search(claim_text or ""))
    quote_negated = bool(_NEGATION_PATTERN.search(quote))
    if overlap and claim_negated != quote_negated:
        return "contradict"
    if overlap:
        return "support"

    if relation in {
        "explicit_mitre_id_match",
        "technique_name_match",
        "claim_span_support",
        "contextual_support",
        "self_referential_source_support",
        "source_support",
    }:
        return "support"
    return "neutral"


def _citation_core_row(citation: dict[str, Any], *, reference_url: str) -> dict[str, Any]:
    return {
        "citation_id": str(citation.get("citation_id") or ""),
        "reference_url": reference_url,
        "method": str(citation.get("method") or ""),
        "relation_to_mitre": str(citation.get("relation_to_mitre") or ""),
        "validation_explanation": str(citation.get("validation_explanation") or ""),
        "start": citation.get("start"),
        "end": citation.get("end"),
        "quote": str(citation.get("quote") or ""),
        "context": str(citation.get("context") or citation.get("snippet") or ""),
    }


def _build_citation_indexes(
    claim_rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    citations_by_id: dict[str, dict[str, Any]] = {}
    citations_by_claim_id: dict[str, list[dict[str, Any]]] = {}

    for row in claim_rows:
        claim = row.get("claim") if isinstance(row.get("claim"), dict) else {}
        claim_id = str(claim.get("claim_id") or "")
        row_citations = row.get("citations") if isinstance(row.get("citations"), list) else []
        if claim_id and claim_id not in citations_by_claim_id:
            citations_by_claim_id[claim_id] = []
        for citation in row_citations:
            if not isinstance(citation, dict):
                continue
            citation_id = str(citation.get("citation_id") or "")
            if not citation_id:
                continue
            if citation_id not in citations_by_id:
                citations_by_id[citation_id] = dict(citation)
            if claim_id:
                existing = {str(c.get("citation_id") or "") for c in citations_by_claim_id[claim_id]}
                if citation_id not in existing:
                    citations_by_claim_id[claim_id].append(dict(citation))

    return citations_by_id, citations_by_claim_id


def _resolved_citations_for_claim(
    claim: dict[str, Any],
    *,
    citations_by_id: dict[str, dict[str, Any]],
    citations_by_claim_id: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    claim_id = str(claim.get("claim_id") or "")
    claim_citation_ids = claim.get("citation_ids") if isinstance(claim.get("citation_ids"), list) else []

    resolved_citations: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for citation_id in claim_citation_ids:
        cid = str(citation_id or "")
        if not cid or cid in seen_ids:
            continue
        seen_ids.add(cid)
        if cid in citations_by_id:
            resolved_citations.append(dict(citations_by_id[cid]))

    if not resolved_citations and claim_id and claim_id in citations_by_claim_id:
        resolved_citations = [dict(c) for c in citations_by_claim_id[claim_id]]
    return resolved_citations


def _normalize_nli_label(label: str) -> str:
    text = str(label or "").strip().lower()
    if text in {"entailment", "entailed", "supports", "support", "supported"}:
        return "entailment"
    if text in {"contradiction", "contradict", "contradicts", "refute", "refutes", "refuted"}:
        return "contradiction"
    if text in {"neutral", "unknown", "uncertain"}:
        return "neutral"
    return ""


def _nli_label_to_stance(label: str) -> str:
    if label == "entailment":
        return "support"
    if label == "contradiction":
        return "contradict"
    return "neutral"


def _build_nli_claim_citation_pairs(
    claims: list[dict[str, Any]],
    claim_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    citations_by_id, citations_by_claim_id = _build_citation_indexes(claim_rows)
    pairs: list[dict[str, Any]] = []
    for claim in claims:
        claim_id = str(claim.get("claim_id") or "")
        if not claim_id:
            continue
        claim_text = str(claim.get("text") or "").strip()
        if not claim_text:
            continue
        for citation in _resolved_citations_for_claim(
            claim,
            citations_by_id=citations_by_id,
            citations_by_claim_id=citations_by_claim_id,
        ):
            citation_id = str(citation.get("citation_id") or "")
            quote = str(citation.get("quote") or "").strip()
            if not citation_id or not quote:
                continue
            pairs.append(
                {
                    "claim_id": claim_id,
                    "citation_id": citation_id,
                    "claim_text": claim_text,
                    "citation_quote": quote,
                    "citation_context": str(citation.get("context") or citation.get("snippet") or ""),
                }
            )
    return pairs


def _nli_stance_labels_for_pairs(
    pairs: list[dict[str, Any]],
    *,
    openai_api_key: str | None = None,
    model_id: str | None = None,
    timeout: int = 60,
    batch_size: int = 24,
) -> tuple[dict[tuple[str, str], dict[str, Any]], str | None, dict[str, int]]:
    if not pairs:
        return {}, None, _zero_token_usage()

    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {}, "nli skipped: missing OPENAI_API_KEY", _zero_token_usage()

    try:
        import openai
    except Exception:
        return {}, "nli skipped: openai package unavailable", _zero_token_usage()

    labels: dict[tuple[str, str], dict[str, Any]] = {}
    errors: list[str] = []
    usage_total = _zero_token_usage()
    resolved_model = model_id or os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)

    try:
        client = openai.OpenAI(api_key=api_key, timeout=timeout)
    except Exception as exc:
        return {}, f"nli client init failed: {type(exc).__name__}", _zero_token_usage()

    for start in range(0, len(pairs), max(1, int(batch_size))):
        batch = pairs[start : start + max(1, int(batch_size))]
        payload = [
            {
                "claim_id": str(pair.get("claim_id") or ""),
                "citation_id": str(pair.get("citation_id") or ""),
                "claim": str(pair.get("claim_text") or "")[:700],
                "citation": str(pair.get("citation_quote") or "")[:900],
                "context": str(pair.get("citation_context") or "")[:900],
            }
            for pair in batch
        ]
        try:
            response = client.chat.completions.create(
                model=resolved_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an NLI classifier. For each pair, classify the citation against the claim as one label: "
                            "entailment, contradiction, or neutral. Return strict JSON with key 'results', where each item has "
                            "claim_id, citation_id, label, confidence (0..1), reason."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps({"pairs": payload}, ensure_ascii=True),
                    },
                ],
                temperature=0,
                response_format={"type": "json_object"},
                max_tokens=max(300, min(2400, 130 * len(batch))),
            )
            usage_total = _merge_token_usage(usage_total, _extract_openai_token_usage(response))
            content = ""
            if response and getattr(response, "choices", None):
                first_choice = response.choices[0]
                message = getattr(first_choice, "message", None)
                content = str(getattr(message, "content", "") or "")
            parsed = json.loads(content or "{}")
            results = parsed.get("results", []) if isinstance(parsed, dict) else []
            for item in results:
                if not isinstance(item, dict):
                    continue
                claim_id = str(item.get("claim_id") or "").strip()
                citation_id = str(item.get("citation_id") or "").strip()
                label = _normalize_nli_label(str(item.get("label") or ""))
                if not claim_id or not citation_id or label not in _NLI_LABEL_CANONICAL:
                    continue
                confidence_raw = item.get("confidence")
                try:
                    confidence = float(confidence_raw)
                except Exception:
                    confidence = None
                labels[(claim_id, citation_id)] = {
                    "label": label,
                    "stance": _nli_label_to_stance(label),
                    "confidence": confidence,
                    "reason": str(item.get("reason") or item.get("explanation") or ""),
                }
        except Exception as exc:
            errors.append(type(exc).__name__)
            continue

    if not labels and errors:
        return {}, f"nli failed: {errors[-1]}", usage_total
    if labels and errors:
        return labels, f"nli partial: labeled={len(labels)}/{len(pairs)} last_error={errors[-1]}", usage_total
    if not labels:
        return {}, "nli returned no labels", usage_total
    return labels, None, usage_total


def _build_claim_consolidation_tables(
    claims: list[dict[str, Any]],
    claim_rows: list[dict[str, Any]],
    *,
    reference_url: str,
    nli_stance_labels: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    citations_by_id, citations_by_claim_id = _build_citation_indexes(claim_rows)
    stance_overrides = nli_stance_labels or {}

    claim_assessments: list[dict[str, Any]] = []
    claim_associations: list[dict[str, Any]] = []

    for claim in claims:
        claim_id = str(claim.get("claim_id") or "")
        claim_text = str(claim.get("text") or "")
        assessment = claim.get("assessment") if isinstance(claim.get("assessment"), dict) else {}
        claim_citation_ids = claim.get("citation_ids") if isinstance(claim.get("citation_ids"), list) else []

        resolved_citations = _resolved_citations_for_claim(
            claim,
            citations_by_id=citations_by_id,
            citations_by_claim_id=citations_by_claim_id,
        )

        support_rows: list[dict[str, Any]] = []
        contradict_rows: list[dict[str, Any]] = []
        neutral_rows: list[dict[str, Any]] = []

        for citation in resolved_citations:
            base = _citation_core_row(citation, reference_url=reference_url)
            citation_id = str(base.get("citation_id") or "")
            nli_entry = stance_overrides.get((claim_id, citation_id), {})
            stance = str(nli_entry.get("stance") or "")
            if stance not in ASSOCIATION_STANCES:
                stance = normalize_association_stance(_citation_stance_for_claim(claim_text, citation))
            stance_source = "nli" if nli_entry else "heuristic"
            association_score = _association_score_for_citation(citation, stance)
            ref_row = {
                "claim_id": claim_id,
                "claim_text": claim_text,
                "association_id": citation_id,
                "stance": normalize_association_stance(stance),
                "association_score": round(float(association_score), 3),
                "stance_source": stance_source,
                "nli_label": str(nli_entry.get("label") or ""),
                "nli_confidence": nli_entry.get("confidence"),
                "nli_reason": str(nli_entry.get("reason") or ""),
                "source": str(claim.get("source") or ""),
                **base,
            }
            claim_associations.append(ref_row)
            if stance == "support":
                support_rows.append(ref_row)
            elif stance == "contradict":
                contradict_rows.append(ref_row)
            else:
                neutral_rows.append(ref_row)

        claim_assessments.append(
            {
                "claim_id": claim_id,
                "claim_text": claim_text,
                "source": str(claim.get("source") or ""),
                "claim_scope": str(claim.get("claim_scope") or ""),
                "falsifiable": bool(claim.get("falsifiable", True)),
                "mitre_id": str(claim.get("mitre_id") or ""),
                "technique_name": str(claim.get("technique_name") or ""),
                "tactic": str(claim.get("tactic") or ""),
                "validation_method": str(claim.get("validation_method") or ""),  # legacy label retained for extraction provenance
                "state": normalize_claim_state(str(assessment.get("state") or "")),
                "confidence": float(assessment.get("confidence") or 0.0),
                "support_score": float(assessment.get("support_score") or 0.0),
                "contradict_score": float(assessment.get("contradict_score") or 0.0),
                "support_count": int(assessment.get("support_count") or 0),
                "contradict_count": int(assessment.get("contradict_count") or 0),
                "cite_count": int(assessment.get("cite_count") or 0),
                "reason": str(assessment.get("reason") or ""),
                "citation_ids": [str(cid) for cid in claim_citation_ids],
                "support_citation_ids": [str(r["citation_id"]) for r in support_rows],
                "contradict_citation_ids": [str(r["citation_id"]) for r in contradict_rows],
                "neutral_citation_ids": [str(r["citation_id"]) for r in neutral_rows],
                "support_reference_count": len(support_rows),
                "contradict_reference_count": len(contradict_rows),
                "neutral_reference_count": len(neutral_rows),
                "support_references": support_rows,
                "contradict_references": contradict_rows,
                "neutral_references": neutral_rows,
                "reference_url": reference_url,
            }
        )

    claim_assessments.sort(
        key=lambda row: (
            STATE_PRIORITY.get(normalize_claim_state(str(row.get("state") or "")), 99),
            -float(row.get("confidence") or 0.0),
            str(row.get("claim_id") or ""),
        )
    )
    return claim_assessments, claim_associations


def _enrich_rows_with_claims(
    document: str,
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    enriched: list[dict[str, Any]] = []
    claims: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()

    for row in rows:
        key = (str(row.get("mitre_id", "")), str(row.get("extraction_text", "")))
        if key in seen_keys:
            continue
        seen_keys.add(key)

        citations = _build_citations_for_row(document, row)
        cite_prefix = "cite_" + hashlib.sha1(
            f"{row.get('source','')}|{row.get('extraction_text','')}|{row.get('mitre_id','')}".encode("utf-8")
        ).hexdigest()[:8]
        for idx, citation in enumerate(citations, start=1):
            citation["citation_id"] = f"{cite_prefix}_{idx}"
            relation_to_mitre, validation_explanation = _citation_relation_and_explanation(row, citation)
            citation["relation_to_mitre"] = relation_to_mitre
            citation["validation_explanation"] = validation_explanation
        assessment = _assess_claim(row, citations)
        claim = _build_claim(row, assessment, citations)

        out_row = dict(row)
        out_row["citations"] = citations
        out_row["claim"] = claim
        out_row["assessment"] = assessment
        enriched.append(out_row)
        claims.append(claim)

    return enriched, claims


def _filter_coherent_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    coherent: list[dict[str, Any]] = []
    for row in rows:
        assessment = row.get("assessment") if isinstance(row.get("assessment"), dict) else {}
        state = normalize_claim_state(str(assessment.get("state") or ""))
        if state not in {"fact", "corroborated", "disputed", "disproven"}:
            continue

        citations = row.get("citations") if isinstance(row.get("citations"), list) else []
        if not citations:
            continue
        if not any(str(c.get("quote") or "").strip() for c in citations if isinstance(c, dict)):
            continue

        # For model-inferred rows, require direct grounding of the ATT&CK mapping.
        if str(row.get("source") or "").startswith("langextract") and str(row.get("mitre_id") or "").strip():
            methods = {
                str(c.get("method") or "")
                for c in citations
                if isinstance(c, dict)
            }
            extraction_matches_technique = _texts_overlap(
                str(row.get("extraction_text") or ""),
                str(row.get("technique_name") or ""),
            )
            if (
                "mitre_id" not in methods
                and "technique_name" not in methods
                and not extraction_matches_technique
            ):
                continue
        coherent.append(row)
    return coherent


def download_attack_bundle(
    *,
    url: str = ATTACK_STIX_URL,
    cache_path: Path = MITRE_CACHE,
    timeout: int = 60,
) -> dict[str, Any]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            cache_path.unlink(missing_ok=True)

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    cache_path.write_text(json.dumps(data))
    return data


def _reference_candidates(bundle: dict[str, Any]) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    for obj in bundle.get("objects", []):
        if obj.get("type") != "attack-pattern" or obj.get("revoked"):
            continue
        mitre_id = _extract_mitre_id(obj)
        if not mitre_id:
            continue
        for ref in obj.get("external_references", []):
            url = str(ref.get("url") or "").strip()
            if not url:
                continue
            refs.append(
                {
                    "mitre_id": mitre_id,
                    "technique_name": str(obj.get("name") or ""),
                    "source_name": str(ref.get("source_name") or "reference"),
                    "url": url,
                }
            )
    return refs


def _domain_is_paywall_prone(url: str) -> bool:
    host = (urlparse(url).netloc or "").lower()
    if not host:
        return False
    return any(host == domain or host.endswith("." + domain) for domain in PAYWALL_DOMAIN_HINTS)


def _response_text_preview(response: requests.Response, *, max_chars: int = 18_000) -> str:
    content_type = (response.headers.get("content-type", "") or "").lower()
    if "html" not in content_type and "text/" not in content_type:
        return ""
    text = response.text or ""
    if not text:
        return ""
    text = re.sub(r"(?is)<(script|style).*?>.*?</\\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return _compact_whitespace(text)[:max_chars]


def _zero_token_usage() -> dict[str, int]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
        "total_tokens": 0,
    }


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _extract_openai_token_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return _zero_token_usage()

    def _lookup(obj: Any, key: str) -> Any:
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    prompt_tokens = _int_or_zero(_lookup(usage, "prompt_tokens"))
    completion_tokens = _int_or_zero(_lookup(usage, "completion_tokens"))
    if prompt_tokens == 0 and completion_tokens == 0:
        # Compatibility with alternate usage field names.
        prompt_tokens = _int_or_zero(_lookup(usage, "input_tokens"))
        completion_tokens = _int_or_zero(_lookup(usage, "output_tokens"))
    total_tokens = _int_or_zero(_lookup(usage, "total_tokens"))
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens

    prompt_details = _lookup(usage, "prompt_tokens_details")
    cached_tokens = _int_or_zero(_lookup(prompt_details, "cached_tokens"))
    if cached_tokens <= 0:
        cached_tokens = _int_or_zero(_lookup(usage, "cached_tokens"))

    return {
        "input_tokens": max(0, prompt_tokens),
        "output_tokens": max(0, completion_tokens),
        "cached_tokens": max(0, cached_tokens),
        "total_tokens": max(0, total_tokens),
    }


def _merge_token_usage(total: dict[str, int], delta: dict[str, int] | None) -> dict[str, int]:
    if not isinstance(total, dict):
        total = _zero_token_usage()
    add = delta if isinstance(delta, dict) else {}
    for key in ("input_tokens", "output_tokens", "cached_tokens", "total_tokens"):
        total[key] = _int_or_zero(total.get(key)) + _int_or_zero(add.get(key))
    if total.get("total_tokens", 0) <= 0:
        total["total_tokens"] = _int_or_zero(total.get("input_tokens")) + _int_or_zero(total.get("output_tokens"))
    return total


def _usage_from_fetch_attempts(attempts: list[dict[str, Any]]) -> dict[str, int]:
    total = _zero_token_usage()
    for attempt in attempts or []:
        if not isinstance(attempt, dict):
            continue
        total = _merge_token_usage(total, attempt.get("token_usage"))
    return total


def _heuristic_paywall_check(url: str, response: requests.Response) -> tuple[bool, str]:
    text = _response_text_preview(response)
    if not text:
        return False, ""

    lower = text.lower()
    hits = [hint for hint in PAYWALL_TEXT_HINTS if hint in lower]
    if len(hits) >= 2:
        return True, f"paywall markers detected ({', '.join(hits[:2])})"
    if _domain_is_paywall_prone(url) and hits:
        return True, f"paywall-prone domain with marker ({hits[0]})"
    if _domain_is_paywall_prone(url) and len(lower) < 900:
        return True, "paywall-prone domain with unusually short article body"
    return False, ""


def _llm_paywall_check(
    text: str,
    *,
    openai_api_key: str | None = None,
    model_id: str | None = None,
) -> tuple[bool, str, dict[str, int]]:
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False, "llm paywall check skipped: missing OPENAI_API_KEY", _zero_token_usage()
    try:
        import openai

        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model_id or os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify whether this page content appears blocked by a paywall. "
                        "Return strict JSON with keys: paywalled (boolean), reason (string)."
                    ),
                },
                {
                    "role": "user",
                    "content": text[:6000],
                },
            ],
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=120,
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)
        paywalled = bool(parsed.get("paywalled"))
        reason = str(parsed.get("reason") or "")
        return paywalled, reason, _extract_openai_token_usage(response)
    except Exception as exc:
        return False, f"llm paywall check failed: {type(exc).__name__}", _zero_token_usage()


def _is_response_paywalled(
    url: str,
    response: requests.Response,
    *,
    llm_paywall_check: bool = False,
    openai_api_key: str | None = None,
    model_id: str | None = None,
) -> tuple[bool, str, dict[str, int]]:
    heur_paywalled, heur_reason = _heuristic_paywall_check(url, response)
    if heur_paywalled:
        return True, heur_reason, _zero_token_usage()
    if not llm_paywall_check:
        return False, "", _zero_token_usage()
    text = _response_text_preview(response)
    if not text:
        return False, "llm paywall check skipped: non-text response", _zero_token_usage()
    llm_paywalled, llm_reason, llm_usage = _llm_paywall_check(
        text,
        openai_api_key=openai_api_key,
        model_id=model_id,
    )
    return llm_paywalled, llm_reason, llm_usage


def _coerce_paywall_result(result: Any) -> tuple[bool, str, dict[str, int]]:
    if isinstance(result, tuple):
        if len(result) >= 3:
            return bool(result[0]), str(result[1] or ""), result[2] if isinstance(result[2], dict) else _zero_token_usage()
        if len(result) == 2:
            return bool(result[0]), str(result[1] or ""), _zero_token_usage()
        if len(result) == 1:
            return bool(result[0]), "", _zero_token_usage()
    if isinstance(result, bool):
        return bool(result), "", _zero_token_usage()
    return False, "", _zero_token_usage()


def _classify_fetch_exception(exc: Exception) -> str:
    if isinstance(exc, requests.Timeout):
        return "timeout"
    if isinstance(exc, requests.HTTPError):
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status == 404:
            return "404"
        if status is not None:
            return f"http_{status}"
        return "http_error"
    if isinstance(exc, requests.ConnectionError):
        return "connection_error"
    if isinstance(exc, requests.TooManyRedirects):
        return "redirect_error"
    return "fetch_error"


def _primary_failure_reason(attempts: list[dict[str, Any]]) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    for attempt in attempts:
        reason = str(attempt.get("failure_reason") or "").strip().lower()
        if not reason:
            continue
        counts[reason] = counts.get(reason, 0) + 1
    if not counts:
        return "unknown", {}
    if len(counts) == 1:
        return next(iter(counts.keys())), counts

    priority = {
        "paywall": 0,
        "404": 1,
        "timeout": 2,
        "connection_error": 3,
        "redirect_error": 4,
        "http_error": 5,
        "fetch_error": 6,
    }
    best = sorted(
        counts.items(),
        key=lambda item: (-item[1], priority.get(item[0], 99), item[0]),
    )[0][0]
    return best, counts


def _fetch_or_raise(url: str, *, user_agent: str, timeout: int = 60) -> requests.Response:
    response = requests.get(url, timeout=timeout, headers={"User-Agent": user_agent})
    response.raise_for_status()
    if not response.content:
        raise RuntimeError(f"empty response body: {url}")
    return response


def _wayback_candidates(url: str, *, user_agent: str, timeout: int = 60) -> list[str]:
    candidates: list[str] = []
    started = time.monotonic()
    LOGGER.info("Wayback lookup start: url=%s timeout=%ss", url, timeout)

    try:
        from waybackpy import WaybackMachineCDXServerAPI

        waybackpy_timeout = max(1, min(int(timeout), DEFAULT_WAYBACKPY_TIMEOUT_SECONDS))

        def _resolve_newest() -> str:
            newest_obj = WaybackMachineCDXServerAPI(url, user_agent).newest()
            return str(getattr(newest_obj, "archive_url", "") or "")

        LOGGER.info("Wayback waybackpy lookup start: url=%s timeout=%ss", url, waybackpy_timeout)
        wp_started = time.monotonic()
        newest = ""
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_resolve_newest)
        try:
            try:
                newest = future.result(timeout=waybackpy_timeout)
            except concurrent.futures.TimeoutError:
                LOGGER.warning(
                    "Wayback waybackpy lookup timed out: url=%s timeout=%ss",
                    url,
                    waybackpy_timeout,
                )
                future.cancel()
            except Exception as exc:
                LOGGER.warning(
                    "Wayback waybackpy lookup failed: url=%s error=%s",
                    url,
                    type(exc).__name__,
                )
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        LOGGER.info(
            "Wayback waybackpy lookup end: url=%s elapsed=%.2fs has_newest=%s",
            url,
            time.monotonic() - wp_started,
            bool(newest),
        )
        if newest:
            candidates.append(newest)
    except Exception as exc:
        LOGGER.debug("Wayback waybackpy unavailable: %s", type(exc).__name__)

    try:
        LOGGER.info("Wayback CDX lookup start: url=%s timeout=%ss", url, timeout)
        cdx_started = time.monotonic()
        cdx_response = requests.get(
            "https://web.archive.org/cdx/search/cdx",
            params={
                "url": url,
                "output": "json",
                "fl": "timestamp,original,statuscode",
                "filter": "statuscode:200",
                "limit": "20",
                "from": "1996",
                "to": "2035",
            },
            timeout=timeout,
            headers={"User-Agent": user_agent},
        )
        cdx_response.raise_for_status()
        rows = cdx_response.json()
        cdx_added = 0
        for row in rows[1:]:
            timestamp = row[0]
            original = row[1]
            candidates.append(f"https://web.archive.org/web/{timestamp}id_/{original}")
            candidates.append(f"https://web.archive.org/web/{timestamp}/{original}")
            cdx_added += 2
        LOGGER.info(
            "Wayback CDX lookup end: url=%s elapsed=%.2fs rows=%d candidates_added=%d",
            url,
            time.monotonic() - cdx_started,
            max(0, len(rows) - 1),
            cdx_added,
        )
    except Exception as exc:
        LOGGER.warning("Wayback CDX lookup failed: url=%s error=%s", url, type(exc).__name__)

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            deduped.append(candidate)
    LOGGER.info(
        "Wayback lookup end: url=%s elapsed=%.2fs candidates_total=%d candidates_deduped=%d",
        url,
        time.monotonic() - started,
        len(candidates),
        len(deduped),
    )
    return deduped


def download_random_reference(
    bundle: dict[str, Any],
    *,
    output_dir: Path = Path("data/mitre_reference_docs"),
    user_agent: str = DEFAULT_USER_AGENT,
    max_candidates: int = 40,
    max_wayback_urls: int = 30,
    prefer_wayback: bool = False,
    llm_paywall_check: bool = True,
    openai_api_key: str | None = None,
    paywall_llm_model_id: str | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    candidates = _reference_candidates(bundle)
    if not candidates:
        raise RuntimeError("no MITRE external references found")

    preferred = [entry for entry in candidates if "attack.mitre.org" not in entry["url"]]
    pool = preferred or candidates
    random.shuffle(pool)
    LOGGER.info(
        "Selecting random MITRE reference (candidate_pool=%d, max_candidates=%d, prefer_wayback=%s, max_wayback_urls=%d)",
        len(pool),
        max_candidates,
        prefer_wayback,
        max_wayback_urls,
    )

    winner: tuple[dict[str, str], requests.Response, str, str] | None = None
    winner_wayback_available: list[str] = []
    winner_wayback_considered: list[str] = []
    winner_attempts: list[dict[str, Any]] = []
    all_attempts: list[dict[str, Any]] = []
    errors: list[str] = []
    for selected in pool[:max_candidates]:
        target_url = selected["url"]
        wayback_urls_all = _wayback_candidates(target_url, user_agent=user_agent, timeout=timeout)
        wayback_urls = wayback_urls_all[:max_wayback_urls]
        LOGGER.info(
            "Wayback options resolved: target=%s available=%d considered=%d",
            target_url,
            len(wayback_urls_all),
            len(wayback_urls),
        )
        if wayback_urls:
            LOGGER.info("Wayback options (considered): %s", wayback_urls)
        else:
            LOGGER.info("No Wayback options found for target=%s", target_url)
        try_wayback_first = prefer_wayback or _domain_is_paywall_prone(target_url)
        candidates: list[tuple[str, str]] = []
        if try_wayback_first:
            candidates.extend([("wayback", wb_url) for wb_url in wayback_urls])
            candidates.append(("direct", target_url))
        else:
            candidates.append(("direct", target_url))
            candidates.extend([("wayback", wb_url) for wb_url in wayback_urls])
        LOGGER.info("Candidate walk order for target=%s: %s", target_url, [url for _, url in candidates])

        attempt_trace: list[dict[str, Any]] = []
        for source, candidate_url in candidates:
            attempt_started = time.monotonic()
            LOGGER.info("Trying candidate: source=%s url=%s", source, candidate_url)
            try:
                response = _fetch_or_raise(candidate_url, user_agent=user_agent, timeout=timeout)
            except Exception as fetch_error:
                reason = _classify_fetch_exception(fetch_error)
                errors.append(f"{candidate_url} -> {reason}")
                LOGGER.debug("Failed candidate fetch: %s (%s)", candidate_url, reason)
                attempt = {
                    "target_url": target_url,
                    "source": source,
                    "url": candidate_url,
                    "status": "fetch_error",
                    "failure_reason": reason,
                    "error": type(fetch_error).__name__,
                    "elapsed_seconds": round(time.monotonic() - attempt_started, 3),
                }
                attempt_trace.append(attempt)
                all_attempts.append(attempt)
                continue
            paywall_result = _is_response_paywalled(
                target_url,
                response,
                llm_paywall_check=llm_paywall_check,
                openai_api_key=openai_api_key,
                model_id=paywall_llm_model_id,
            )
            paywalled, paywall_reason, paywall_usage = _coerce_paywall_result(paywall_result)
            if paywalled:
                errors.append(f"{candidate_url} -> paywall ({paywall_reason})")
                LOGGER.info("Skipped paywalled candidate: %s (%s)", candidate_url, paywall_reason)
                attempt = {
                    "target_url": target_url,
                    "source": source,
                    "url": candidate_url,
                    "status": "paywalled",
                    "failure_reason": "paywall",
                    "reason": paywall_reason,
                    "elapsed_seconds": round(time.monotonic() - attempt_started, 3),
                    "token_usage": paywall_usage,
                }
                attempt_trace.append(attempt)
                all_attempts.append(attempt)
                continue
            winner = (selected, response, source, candidate_url)
            winner_wayback_available = wayback_urls_all
            winner_wayback_considered = wayback_urls
            attempt = {
                "target_url": target_url,
                "source": source,
                "url": candidate_url,
                "status": "ok",
                "elapsed_seconds": round(time.monotonic() - attempt_started, 3),
                "token_usage": paywall_usage,
            }
            attempt_trace.append(attempt)
            all_attempts.append(attempt)
            winner_attempts = attempt_trace
            LOGGER.info("Selected candidate: source=%s url=%s", source, candidate_url)
            break
        if winner is not None:
            break

    if winner is None:
        failure_reason, reason_counts = _primary_failure_reason(all_attempts)
        raise ReferenceFetchError(
            "failed to download a usable MITRE reference; "
            f"failure_reason={failure_reason}; reason_counts={reason_counts}; "
            "sample errors:\n" + "\n".join(errors[:5]),
            failure_reason=failure_reason,
            attempts=all_attempts,
        )

    selected, response, source, resolved_url = winner
    output_dir.mkdir(parents=True, exist_ok=True)
    content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
    content_suffix = {
        "text/html": ".html",
        "text/plain": ".txt",
        "application/pdf": ".pdf",
        "application/json": ".json",
    }.get(content_type)
    if not content_suffix:
        parsed = urlparse(resolved_url)
        content_suffix = Path(parsed.path).suffix or ".bin"

    safe_base = re.sub(
        r"[^a-zA-Z0-9._-]+",
        "_",
        f"{selected['mitre_id']}_{selected['source_name']}",
    ).strip("_")
    output_path = output_dir / f"{safe_base}{content_suffix}"
    output_path.write_bytes(response.content)
    LOGGER.info(
        "Downloaded reference: source=%s url=%s content_type=%s bytes=%d",
        source,
        resolved_url,
        content_type or "unknown",
        len(response.content),
    )
    return {
        "meta": selected,
        "path": str(output_path),
        "content_type": content_type,
        "resolved_url": resolved_url,
        "download_source": source,
        "bytes": len(response.content),
        "wayback_candidates_available": winner_wayback_available,
        "wayback_candidates_considered": winner_wayback_considered,
        "fetch_attempts": winner_attempts,
    }


def download_reference_from_url(
    url: str,
    *,
    output_dir: Path = Path("data/mitre_reference_docs"),
    user_agent: str = DEFAULT_USER_AGENT,
    max_wayback_urls: int = 30,
    prefer_wayback: bool = False,
    llm_paywall_check: bool = True,
    openai_api_key: str | None = None,
    paywall_llm_model_id: str | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    target_url = str(url or "").strip()
    if not target_url:
        raise RuntimeError("reference URL is empty")
    parsed_target = urlparse(target_url)
    if parsed_target.scheme not in {"http", "https"}:
        raise RuntimeError("reference URL must start with http:// or https://")
    LOGGER.info(
        "Fetching explicit reference URL: %s (prefer_wayback=%s, max_wayback_urls=%d)",
        target_url,
        prefer_wayback,
        max_wayback_urls,
    )

    winner: tuple[requests.Response, str, str] | None = None
    errors: list[str] = []
    wayback_urls_all = _wayback_candidates(target_url, user_agent=user_agent, timeout=timeout)
    wayback_urls = wayback_urls_all[:max_wayback_urls]
    LOGGER.info(
        "Wayback options resolved: target=%s available=%d considered=%d",
        target_url,
        len(wayback_urls_all),
        len(wayback_urls),
    )
    if wayback_urls:
        LOGGER.info("Wayback options (considered): %s", wayback_urls)
    else:
        LOGGER.info("No Wayback options found for target=%s", target_url)
    try_wayback_first = prefer_wayback or _domain_is_paywall_prone(target_url)
    candidates: list[tuple[str, str]] = []
    if try_wayback_first:
        candidates.extend([("wayback", wb_url) for wb_url in wayback_urls])
        candidates.append(("direct", target_url))
    else:
        candidates.append(("direct", target_url))
        candidates.extend([("wayback", wb_url) for wb_url in wayback_urls])
    LOGGER.info("Candidate walk order for target=%s: %s", target_url, [url for _, url in candidates])

    attempt_trace: list[dict[str, Any]] = []
    for source, candidate_url in candidates:
        attempt_started = time.monotonic()
        LOGGER.info("Trying candidate: source=%s url=%s", source, candidate_url)
        try:
            response = _fetch_or_raise(candidate_url, user_agent=user_agent, timeout=timeout)
        except Exception as fetch_error:
            reason = _classify_fetch_exception(fetch_error)
            errors.append(f"{candidate_url} -> {reason}")
            LOGGER.debug("Failed candidate fetch: %s (%s)", candidate_url, reason)
            attempt_trace.append(
                {
                    "source": source,
                    "url": candidate_url,
                    "status": "fetch_error",
                    "failure_reason": reason,
                    "error": type(fetch_error).__name__,
                    "elapsed_seconds": round(time.monotonic() - attempt_started, 3),
                }
            )
            continue
        paywall_result = _is_response_paywalled(
            target_url,
            response,
            llm_paywall_check=llm_paywall_check,
            openai_api_key=openai_api_key,
            model_id=paywall_llm_model_id,
        )
        paywalled, paywall_reason, paywall_usage = _coerce_paywall_result(paywall_result)
        if paywalled:
            errors.append(f"{candidate_url} -> paywall ({paywall_reason})")
            LOGGER.info("Skipped paywalled candidate: %s (%s)", candidate_url, paywall_reason)
            attempt_trace.append(
                {
                    "source": source,
                    "url": candidate_url,
                    "status": "paywalled",
                    "failure_reason": "paywall",
                    "reason": paywall_reason,
                    "elapsed_seconds": round(time.monotonic() - attempt_started, 3),
                    "token_usage": paywall_usage,
                }
            )
            continue
        winner = (response, source, candidate_url)
        attempt_trace.append(
            {
                "source": source,
                "url": candidate_url,
                "status": "ok",
                "elapsed_seconds": round(time.monotonic() - attempt_started, 3),
                "token_usage": paywall_usage,
            }
        )
        LOGGER.info("Selected candidate: source=%s url=%s", source, candidate_url)
        break

    if winner is None:
        failure_reason, reason_counts = _primary_failure_reason(attempt_trace)
        raise ReferenceFetchError(
            "failed to download reference URL; "
            f"failure_reason={failure_reason}; reason_counts={reason_counts}; "
            "sample errors:\n" + "\n".join(errors[:5]),
            failure_reason=failure_reason,
            attempts=attempt_trace,
        )

    response, source, resolved_url = winner
    output_dir.mkdir(parents=True, exist_ok=True)
    content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
    content_suffix = {
        "text/html": ".html",
        "text/plain": ".txt",
        "application/pdf": ".pdf",
        "application/json": ".json",
    }.get(content_type)
    if not content_suffix:
        parsed = urlparse(resolved_url)
        content_suffix = Path(parsed.path).suffix or ".bin"

    parsed = urlparse(target_url)
    source_name = Path(parsed.path).stem or parsed.netloc or "reference_url"
    source_name = re.sub(r"[_-]+", " ", source_name).strip() or "reference_url"
    url_ids = MITRE_ID_PATTERN.findall(target_url)
    safe_base = re.sub(
        r"[^a-zA-Z0-9._-]+",
        "_",
        f"url_{parsed.netloc}_{Path(parsed.path).stem or 'reference'}",
    ).strip("_")
    output_path = output_dir / f"{safe_base}{content_suffix}"
    output_path.write_bytes(response.content)
    LOGGER.info(
        "Downloaded explicit reference: source=%s url=%s content_type=%s bytes=%d",
        source,
        resolved_url,
        content_type or "unknown",
        len(response.content),
    )
    return {
        "meta": {
            "mitre_id": url_ids[0] if url_ids else "",
            "technique_name": "",
            "source_name": source_name,
            "url": target_url,
        },
        "path": str(output_path),
        "content_type": content_type,
        "resolved_url": resolved_url,
        "download_source": source,
        "bytes": len(response.content),
        "wayback_candidates_available": wayback_urls_all,
        "wayback_candidates_considered": wayback_urls,
        "fetch_attempts": attempt_trace,
    }


def _compact_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text or "")
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def load_reference_text(reference_download: dict[str, Any], *, max_chars: int = 120_000) -> str:
    download_path = Path(str(reference_download["path"]))
    download_type = str(reference_download.get("content_type") or "").lower()

    document = ""
    if download_path.suffix.lower() == ".txt" or download_type.startswith("text/plain"):
        document = download_path.read_text(errors="ignore")
    elif download_path.suffix.lower() == ".json" or download_type == "application/json":
        raw_json = download_path.read_text(errors="ignore")
        try:
            document = json.dumps(json.loads(raw_json), indent=2)
        except Exception:
            document = raw_json
    elif download_path.suffix.lower() == ".html" or "html" in download_type:
        raw_html = download_path.read_text(errors="ignore")
        try:
            import trafilatura

            extracted = trafilatura.extract(raw_html, output_format="txt")
            document = extracted if extracted and extracted.strip() else raw_html
        except Exception:
            document = raw_html
        document = re.sub(r"(?is)<(script|style).*?>.*?</\\1>", " ", document)
        document = re.sub(r"(?s)<[^>]+>", " ", document)
        document = document.replace("&nbsp;", " ").replace("&amp;", "&")
    elif download_path.suffix.lower() == ".pdf" or download_type == "application/pdf":
        try:
            import pypdf

            reader = pypdf.PdfReader(str(download_path))
            document = "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as exc:
            raise RuntimeError(f"PDF text extraction failed for {download_path}: {exc}") from exc
    else:
        document = download_path.read_text(errors="ignore")

    return _compact_whitespace(document)[:max_chars]


def _maybe_import_langextract() -> Any | None:
    try:
        import langextract as lx

        return lx
    except Exception:
        return None


def _coerce_example(lx_module: Any, text: str, raw_extractions: list[dict[str, Any]]) -> Any:
    if lx_module is None:
        return {"text": text, "extractions": raw_extractions}
    try:
        extraction_cls = lx_module.data.Extraction
        example_cls = lx_module.data.ExampleData
        return example_cls(
            text=text,
            extractions=[
                extraction_cls(
                    extraction_class=row["extraction_class"],
                    extraction_text=row["extraction_text"],
                    attributes=row["attributes"],
                )
                for row in raw_extractions
            ],
        )
    except Exception:
        return {"text": text, "extractions": raw_extractions}


def _build_examples(
    lx_module: Any | None,
    *,
    example_texts: list[tuple[str, list[dict[str, Any]]]] | None = None,
) -> list[Any]:
    rows = example_texts if example_texts is not None else DEFAULT_EXAMPLE_TEXTS
    return [_coerce_example(lx_module, text, sample_rows) for text, sample_rows in rows]


def _normalize_langextract_output(
    raw: object,
    *,
    allowed_classes: set[str] | None = None,
) -> list[dict[str, Any]]:
    candidates: Any = []
    if raw is None:
        return []
    if hasattr(raw, "extractions"):
        candidates = raw.extractions
    elif isinstance(raw, dict):
        candidates = raw.get("extractions", raw.get("results", []))
    elif isinstance(raw, list):
        candidates = raw

    out: list[dict[str, Any]] = []
    for item in candidates or []:
        if isinstance(item, dict):
            extraction_class = item.get("extraction_class")
            extraction_text = item.get("extraction_text", "")
            attributes = item.get("attributes", {}) or {}
        else:
            extraction_class = getattr(item, "extraction_class", None)
            extraction_text = getattr(item, "extraction_text", "")
            attributes = getattr(item, "attributes", {}) or {}
        if allowed_classes is not None and extraction_class and extraction_class not in allowed_classes:
            continue
        out.append(
            {
                "extraction_class": extraction_class or "mitre_technique",
                "extraction_text": extraction_text,
                "attributes": attributes if isinstance(attributes, dict) else {},
            }
        )
    return out


def _resolve_extract_signature(extract_fn: Any) -> tuple[bool, set[str]]:
    """Return (accepts_kwargs, accepted_params), unwrapping langextract wrappers."""
    accepts_kwargs = True
    accepted_params: set[str] = set()

    try:
        signature = inspect.signature(extract_fn)
        accepted_params = set(signature.parameters.keys())
        accepts_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        )
    except Exception:
        return True, set()

    # Some langextract versions expose extract(*args, **kwargs) and forward to
    # a strict extract_func. If so, inspect the inner function for real params.
    if accepted_params.issubset({"args", "kwargs"}):
        try:
            module = inspect.getmodule(extract_fn)
            inner = getattr(module, "extract_func", None) if module is not None else None
            if callable(inner):
                inner_sig = inspect.signature(inner)
                accepted_params = set(inner_sig.parameters.keys())
                accepts_kwargs = any(
                    param.kind == inspect.Parameter.VAR_KEYWORD
                    for param in inner_sig.parameters.values()
                )
        except Exception:
            pass

    return accepts_kwargs, accepted_params


def run_langextract(
    document: str,
    *,
    prompt: str = DEFAULT_PROMPT,
    example_texts: list[tuple[str, list[dict[str, Any]]]] | None = None,
    allowed_extraction_classes: set[str] | None = None,
    model_id: str | None = None,
    provider: str = "openai",
    openai_api_key: str | None = None,
    openai_reasoning_effort: str = DEFAULT_OPENAI_REASONING_EFFORT,
    show_progress: bool = False,
    quiet_absl: bool = True,
    lx_module: Any | None = None,
) -> list[dict[str, Any]]:
    active_module = lx_module or _maybe_import_langextract()
    if active_module is None:
        raise RuntimeError(
            "langextract is not installed. Install it in this environment, then rerun."
        )

    extract_fn = getattr(active_module, "extract", None)
    if extract_fn is None:
        raise RuntimeError("langextract.extract is not available")
    resolved_allowed_classes = (
        set(allowed_extraction_classes) if allowed_extraction_classes is not None else {"mitre_technique"}
    )

    provider_normalized = str(provider or "auto").strip().lower()
    if provider_normalized not in {"auto", "openai"}:
        raise RuntimeError(f"unsupported provider: {provider}")

    resolved_model_id = model_id
    provider_kwargs_options: list[dict[str, Any]] = [{}]
    if provider_normalized == "openai":
        resolved_model_id = resolved_model_id or os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        resolved_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "OpenAI mode requires OPENAI_API_KEY (or pass --openai-api-key)."
            )
        # LangExtract recommends these settings for OpenAI models.
        # Prefer schema-constrained output for reliable parsing, then retry with
        # non-schema mode for wider model compatibility.
        openai_base_kwargs: dict[str, Any] = {
            "api_key": resolved_key,
            "fence_output": True,
            "use_schema_constraints": False,
        }
        openai_schema_kwargs: dict[str, Any] = {
            "api_key": resolved_key,
            "fence_output": False,
            "use_schema_constraints": True,
        }
        provider_kwargs_options = [dict(openai_schema_kwargs), dict(openai_base_kwargs)]
        if openai_reasoning_effort:
            # Prefer reasoning effort when available, but keep a no-reasoning
            # fallback for OpenAI SDK/model combinations that reject it.
            provider_kwargs_options.insert(
                0,
                {
                    **openai_schema_kwargs,
                    "language_model_params": {"reasoning_effort": openai_reasoning_effort},
                },
            )
            provider_kwargs_options.insert(
                2,
                {
                    **openai_base_kwargs,
                    "language_model_params": {"reasoning_effort": openai_reasoning_effort},
                },
            )

    examples = _build_examples(active_module, example_texts=example_texts)
    call_variants: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for provider_kwargs in provider_kwargs_options:
        base_kwargs: dict[str, Any] = {
            "examples": examples,
            "show_progress": show_progress,
            **provider_kwargs,
        }
        if resolved_model_id:
            base_kwargs["model_id"] = resolved_model_id
        call_variants.extend(
            [
                ((), {"text_or_documents": document, "prompt_description": prompt, **base_kwargs}),
                ((), {"text": document, "prompt_description": prompt, **base_kwargs}),
                ((), {"text": document, "prompt": prompt, **base_kwargs}),
                ((), {"input_text": document, "prompt": prompt, **base_kwargs}),
                ((), {"input": document, "prompt": prompt, **base_kwargs}),
                ((), {"text_or_documents": document, "prompt": prompt, **base_kwargs}),
                ((document,), {"prompt_description": prompt, **base_kwargs}),
                ((document,), {"prompt": prompt, **base_kwargs}),
            ]
        )

    accepts_kwargs, accepted_params = _resolve_extract_signature(extract_fn)

    last_error: Exception | None = None
    for args, kwargs in call_variants:
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        if not accepts_kwargs:
            kwargs = {key: value for key, value in kwargs.items() if key in accepted_params}
        if not args and not kwargs:
            continue
        try:
            ctx = nullcontext()
            absl_logger = None
            prev_level = None
            if quiet_absl:
                absl_logger = logging.getLogger("absl")
                prev_level = absl_logger.level
                absl_logger.setLevel(logging.CRITICAL)
            with ctx:
                raw = extract_fn(*args, **kwargs)
            if absl_logger is not None and prev_level is not None:
                absl_logger.setLevel(prev_level)
            return _normalize_langextract_output(raw, allowed_classes=resolved_allowed_classes)
        except Exception as exc:
            if quiet_absl:
                logging.getLogger("absl").setLevel(prev_level if prev_level is not None else logging.NOTSET)
            last_error = exc
            continue
    if last_error is None:
        raise RuntimeError("langextract call failed with no usable argument variants")
    raise RuntimeError(f"langextract call failed: {type(last_error).__name__}: {last_error}")


def run_claim_extraction_first_pass(
    document: str,
    *,
    technique_lookup: dict[str, dict[str, str]],
    model_id: str | None = None,
    provider: str = "openai",
    openai_api_key: str | None = None,
    openai_reasoning_effort: str = DEFAULT_OPENAI_REASONING_EFFORT,
    show_progress: bool = False,
    quiet_absl: bool = True,
) -> list[dict[str, Any]]:
    raw_rows = run_langextract(
        document,
        prompt=CLAIM_PASS_PROMPT,
        example_texts=CLAIM_PASS_EXAMPLE_TEXTS,
        allowed_extraction_classes={"falsifiable_claim", "claim", "cybersecurity_claim"},
        model_id=model_id,
        provider=provider,
        openai_api_key=openai_api_key,
        openai_reasoning_effort=openai_reasoning_effort,
        show_progress=show_progress,
        quiet_absl=quiet_absl,
    )
    technique_name_index = _build_technique_name_index(technique_lookup)
    normalized: list[dict[str, Any]] = []
    for row in raw_rows:
        attrs = row.get("attributes") if isinstance(row.get("attributes"), dict) else {}
        claim_text = _compact_whitespace(str(row.get("extraction_text") or ""))
        if not claim_text:
            continue
        if not _is_falsifiable_sentence(claim_text):
            continue

        mitre_id = str(attrs.get("mitre_technique_id") or "").strip()
        technique_name = ""
        tactic = ""
        match_method = ""
        if mitre_id and mitre_id in technique_lookup:
            matched = technique_lookup.get(mitre_id, {})
            technique_name = str(matched.get("name") or "")
            tactic = str(matched.get("tactic") or "")
            match_method = "llm_attribute_mitre_id"
        else:
            mapped_id, mapped_name, mapped_tactic, mapped_method = _map_sentence_to_mitre(
                claim_text,
                technique_lookup,
                technique_name_index,
            )
            mitre_id = mapped_id
            technique_name = mapped_name
            tactic = mapped_tactic
            match_method = mapped_method

        claim_scope = str(attrs.get("claim_scope") or "").strip().lower()
        if claim_scope not in {"cybersecurity", "general"}:
            claim_scope = "cybersecurity" if (_is_cybersecurity_sentence(claim_text) or mitre_id) else "general"
        validation_method = str(attrs.get("validation_method") or "").strip().lower()
        if validation_method not in {"mitre_data", "self_referential"}:
            validation_method = "mitre_data" if (claim_scope == "cybersecurity" and mitre_id) else "self_referential"

        confidence = str(attrs.get("confidence") or "medium").lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = "medium"

        normalized.append(
            {
                "mitre_id": mitre_id,
                "technique_name": technique_name,
                "tactic": tactic,
                "extraction_text": claim_text,
                "claim_text": claim_text,
                "confidence": confidence,
                "source": "langextract_claim_first_pass",
                "claim_scope": claim_scope,
                "validation_method": validation_method,
                "mitre_match_method": match_method,
                "raw": row,
            }
        )
    return normalized


def _dedupe_claim_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        text = _compact_whitespace(str(row.get("claim_text") or row.get("extraction_text") or ""))
        mitre_id = str(row.get("mitre_id") or "")
        key = (text.lower(), mitre_id)
        if not text or key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _summarize_claim_assessments(claims: list[dict[str, Any]]) -> dict[str, int]:
    rows = [
        {"state": normalize_claim_state(str((claim.get("assessment") or {}).get("state") or ""))}
        for claim in claims
    ]
    return summarize_claim_states(rows)


def _build_mitre_linkages(enriched_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    linkages: list[dict[str, Any]] = []
    for row in enriched_rows:
        mitre_id = str(row.get("mitre_id") or "").strip()
        if not mitre_id:
            continue
        claim = row.get("claim") if isinstance(row.get("claim"), dict) else {}
        linkages.append(
            {
                "claim_id": claim.get("claim_id"),
                "claim_text": claim.get("text"),
                "mitre_id": mitre_id,
                "technique_name": row.get("technique_name", ""),
                "tactic": row.get("tactic", ""),
                "validation_method": claim.get("validation_method") or row.get("validation_method"),
                "assessment": row.get("assessment"),
                "citations": row.get("citations", []),
                "source": row.get("source", ""),
            }
        )
    return linkages


def _dedupe_claim_objects(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for claim in claims:
        text = _compact_whitespace(str(claim.get("text") or ""))
        mitre_id = str(claim.get("mitre_id") or "")
        key = (text.lower(), mitre_id)
        if not text or key in seen:
            continue
        seen.add(key)
        out.append(claim)
    return out


def _sort_claim_objects(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def _key(claim: dict[str, Any]) -> tuple[int, float, str]:
        mitre_rank = 0 if str(claim.get("mitre_id") or "").strip() else 1
        score = float(claim.get("assessment", {}).get("confidence") or 0.0)
        claim_id = str(claim.get("claim_id") or "")
        return (mitre_rank, -score, claim_id)

    return sorted(claims, key=_key)


def run_random_reference_langextract(
    *,
    bundle_url: str = ATTACK_STIX_URL,
    cache_path: str | Path = MITRE_CACHE,
    output_dir: str | Path = "data/mitre_reference_docs",
    reference_url: str | None = None,
    prefer_wayback: bool = True,
    llm_paywall_check: bool = True,
    max_wayback_urls: int = 8,
    model_id: str | None = None,
    provider: str = "openai",
    openai_api_key: str | None = None,
    openai_reasoning_effort: str = DEFAULT_OPENAI_REASONING_EFFORT,
    nli_stance_check: bool = True,
    nli_batch_size: int = 24,
    show_progress: bool = False,
    quiet_absl: bool = True,
    use_langextract: bool = True,
    max_chars: int = 120_000,
    timeout: int = 60,
    prompt: str = DEFAULT_PROMPT,
) -> dict[str, Any]:
    """Download one random MITRE reference (or explicit URL), then run extraction."""
    provider_normalized = str(provider or "auto").strip().lower()
    resolved_status_model = model_id
    if provider_normalized == "openai" and not resolved_status_model:
        resolved_status_model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    LOGGER.info(
        "Starting workflow: provider=%s model=%s reference_url_override=%s",
        provider_normalized,
        resolved_status_model or "",
        bool(reference_url),
    )

    cache = Path(cache_path)
    output = Path(output_dir)
    bundle = download_attack_bundle(url=bundle_url, cache_path=cache, timeout=timeout)
    technique_lookup = _build_technique_lookup(bundle)
    LOGGER.info("Loaded ATT&CK bundle techniques=%d cache=%s", len(technique_lookup), cache)
    if reference_url:
        reference = download_reference_from_url(
            reference_url,
            output_dir=output,
            max_wayback_urls=max_wayback_urls,
            prefer_wayback=prefer_wayback,
            llm_paywall_check=llm_paywall_check,
            openai_api_key=openai_api_key,
            paywall_llm_model_id=resolved_status_model,
            timeout=timeout,
        )
    else:
        reference = download_random_reference(
            bundle,
            output_dir=output,
            max_wayback_urls=max_wayback_urls,
            prefer_wayback=prefer_wayback,
            llm_paywall_check=llm_paywall_check,
            openai_api_key=openai_api_key,
            paywall_llm_model_id=resolved_status_model,
            timeout=timeout,
        )
    document = load_reference_text(reference, max_chars=max_chars)
    LOGGER.info(
        "Loaded reference document chars=%d source=%s",
        len(document),
        reference.get("resolved_url", ""),
    )
    langextract_error: str | None = None
    results_source = "disabled"
    raw_results: list[dict[str, Any]]
    if use_langextract:
        try:
            raw_results = run_langextract(
                document,
                prompt=prompt,
                model_id=model_id,
                provider=provider,
                openai_api_key=openai_api_key,
                openai_reasoning_effort=openai_reasoning_effort,
                show_progress=show_progress,
                quiet_absl=quiet_absl,
            )
            results_source = "langextract"
            LOGGER.info("Technique extraction rows=%d source=langextract", len(raw_results))
        except RuntimeError as exc:
            langextract_error = str(exc)
            raw_results = []
            LOGGER.warning("Technique extraction failed, continuing with fallback paths: %s", exc)
    else:
        raw_results = []
        LOGGER.info("Technique extraction disabled (--no-langextract)")

    normalized_rows = _normalize_extracted_rows(raw_results, technique_lookup)
    mitre_enriched_rows, mitre_claims = _enrich_rows_with_claims(document, normalized_rows)
    coherent_rows = _filter_coherent_rows(mitre_enriched_rows)
    if coherent_rows:
        mitre_enriched_rows = coherent_rows
        mitre_claims = [row.get("claim", {}) for row in mitre_enriched_rows]
        if results_source == "disabled":
            results_source = "langextract"
    else:
        fallback_rows = _normalize_extracted_rows(
            _fallback_extract_from_document(document, technique_lookup),
            technique_lookup,
        )
        fallback_enriched, _ = _enrich_rows_with_claims(document, fallback_rows)
        coherent_fallback = _filter_coherent_rows(fallback_enriched)
        if coherent_fallback:
            mitre_enriched_rows = coherent_fallback
            mitre_claims = [row.get("claim", {}) for row in mitre_enriched_rows]
            results_source = "fallback_id_match"
            LOGGER.info("Using deterministic fallback_id_match rows=%d", len(mitre_enriched_rows))
        else:
            metadata_row = _reference_metadata_row(
                reference,
                technique_lookup,
                document=document,
            )
            if metadata_row is not None:
                mitre_enriched_rows, mitre_claims = _enrich_rows_with_claims(document, [metadata_row])
                results_source = "reference_metadata"
                LOGGER.info("Using reference_metadata fallback rows=%d", len(mitre_enriched_rows))
            else:
                mitre_enriched_rows = []
                mitre_claims = []
                LOGGER.warning("No MITRE rows available after all extraction/fallback paths")

    claim_first_pass_error: str | None = None
    claim_first_pass_rows: list[dict[str, Any]] = []
    if use_langextract:
        try:
            claim_first_pass_rows = run_claim_extraction_first_pass(
                document,
                technique_lookup=technique_lookup,
                model_id=model_id,
                provider=provider,
                openai_api_key=openai_api_key,
                openai_reasoning_effort=openai_reasoning_effort,
                show_progress=show_progress,
                quiet_absl=quiet_absl,
            )
            LOGGER.info("Falsifiable claim first-pass rows=%d", len(claim_first_pass_rows))
        except RuntimeError as exc:
            claim_first_pass_error = str(exc)
            LOGGER.warning("Falsifiable claim first-pass failed: %s", exc)

    deterministic_claim_rows = _extract_falsifiable_claim_rows(document, technique_lookup)
    claim_candidate_rows = _dedupe_claim_rows(claim_first_pass_rows + deterministic_claim_rows)
    claim_enriched_rows, _ = _enrich_rows_with_claims(document, claim_candidate_rows)
    coherent_claim_rows = _filter_coherent_rows(claim_enriched_rows)
    falsifiable_claims = [row.get("claim", {}) for row in coherent_claim_rows]
    if not falsifiable_claims:
        falsifiable_claims = list(mitre_claims)

    combined_claims = _sort_claim_objects(
        _dedupe_claim_objects(falsifiable_claims + list(mitre_claims))
    )
    claim_row_pool = list(mitre_enriched_rows) + list(coherent_claim_rows)
    paywall_usage = _usage_from_fetch_attempts(
        reference.get("fetch_attempts") if isinstance(reference.get("fetch_attempts"), list) else []
    )
    nli_pairs: list[dict[str, Any]] = []
    nli_stance_labels: dict[tuple[str, str], dict[str, Any]] = {}
    nli_status: str | None = None
    nli_usage = _zero_token_usage()
    if nli_stance_check:
        nli_pairs = _build_nli_claim_citation_pairs(combined_claims, claim_row_pool)
        LOGGER.info("NLI stance check start: pairs=%d", len(nli_pairs))
        if nli_pairs:
            nli_stance_labels, nli_status, nli_usage = _nli_stance_labels_for_pairs(
                nli_pairs,
                openai_api_key=openai_api_key,
                model_id=resolved_status_model,
                timeout=timeout,
                batch_size=nli_batch_size,
            )
            if nli_status:
                LOGGER.info("NLI stance check status: %s", nli_status)
        else:
            nli_status = "nli skipped: no claim-citation pairs"
            LOGGER.info("NLI stance check skipped: no claim-citation pairs")
    else:
        nli_status = "nli disabled"
        LOGGER.info("NLI stance check disabled")

    langextract_usage = {
        **_zero_token_usage(),
        "available": False,
        "note": "langextract wrapper does not expose provider token usage",
    }
    token_usage_sections = {
        "paywall_check": paywall_usage,
        "nli_stance": nli_usage,
        "langextract": langextract_usage,
    }
    token_usage_overall = _merge_token_usage(_zero_token_usage(), paywall_usage)
    token_usage_overall = _merge_token_usage(token_usage_overall, nli_usage)
    token_usage = {
        "overall": token_usage_overall,
        "sections": token_usage_sections,
    }
    claim_assessments, claim_associations = _build_claim_consolidation_tables(
        combined_claims,
        claim_row_pool,
        reference_url=str(reference.get("resolved_url") or reference.get("url") or ""),
        nli_stance_labels=nli_stance_labels,
    )
    assessment_summary = {
        "overall": _summarize_claim_assessments(combined_claims),
        "mitre": _summarize_claim_assessments(mitre_claims),
        "falsifiable": _summarize_claim_assessments(falsifiable_claims),
    }
    mitre_linkages = _build_mitre_linkages(
        coherent_claim_rows if coherent_claim_rows else mitre_enriched_rows
    )
    LOGGER.info(
        "Workflow complete: claims=%d falsifiable_claims=%d mitre_linkages=%d results_source=%s",
        len(combined_claims),
        len(falsifiable_claims),
        len(mitre_linkages),
        results_source,
    )

    technique_count = len(technique_lookup)
    return {
        "status": {
            "techniques": technique_count,
            "langextract_available": _maybe_import_langextract() is not None,
            "provider": provider_normalized,
            "model_id": resolved_status_model,
            "openai_reasoning_effort": (
                openai_reasoning_effort if provider_normalized == "openai" else None
            ),
            "show_progress": show_progress,
            "quiet_absl": quiet_absl,
            "results_source": results_source,
            "langextract_error": langextract_error,
            "claim_first_pass_error": claim_first_pass_error,
            "claims_count": len(combined_claims),
            "mitre_claims_count": len(mitre_claims),
            "falsifiable_claims_count": len(falsifiable_claims),
            "mitre_linkages_count": len(mitre_linkages),
            "claim_assessments_count": len(claim_assessments),
            "claim_associations_count": len(claim_associations),
            "cache_path": str(cache),
            "reference_url_override": reference_url or None,
            "prefer_wayback": prefer_wayback,
            "llm_paywall_check": llm_paywall_check,
            "max_wayback_urls": max_wayback_urls,
            "nli_stance_check": nli_stance_check,
            "nli_pairs_count": len(nli_pairs),
            "nli_labeled_count": len(nli_stance_labels),
            "nli_status": nli_status,
            "token_usage_overall": token_usage_overall,
        },
        "reference": reference,
        "document": document,
        "token_usage": token_usage,
        "technique_mentions": mitre_enriched_rows,
        "claims": combined_claims,
        "claim_assessments": claim_assessments,
        "claim_associations": claim_associations,
        "mitre_claims": _sort_claim_objects(_dedupe_claim_objects(mitre_claims)),
        "falsifiable_claims": _sort_claim_objects(_dedupe_claim_objects(falsifiable_claims)),
        "mitre_linkages": mitre_linkages,
        "assessment_summary": assessment_summary,
    }


def to_cli_payload(result: dict[str, Any], *, include_document: bool, preview_chars: int) -> dict[str, Any]:
    payload = dict(result)
    document = str(payload.pop("document", "") or "")
    payload["document_chars"] = len(document)
    if include_document:
        payload["document"] = document
    else:
        payload["document_preview"] = document[: max(0, int(preview_chars))]
    return payload


def _add_extract_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--bundle-url",
        default=ATTACK_STIX_URL,
        help="MITRE ATT&CK enterprise bundle URL.",
    )
    parser.add_argument(
        "--cache-path",
        default=str(MITRE_CACHE),
        help="Path for cached MITRE ATT&CK bundle JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/mitre_reference_docs",
        help="Directory where the fetched reference document is saved.",
    )
    parser.add_argument(
        "--reference-url",
        default=None,
        help="Fetch and process this URL instead of a random MITRE external reference.",
    )
    wayback_group = parser.add_mutually_exclusive_group()
    wayback_group.add_argument(
        "--wayback-first",
        dest="wayback_first",
        action="store_true",
        help="Prefer Wayback snapshots before direct fetch (default).",
    )
    wayback_group.add_argument(
        "--direct-first",
        dest="wayback_first",
        action="store_false",
        help="Try direct fetch before Wayback snapshots.",
    )
    parser.set_defaults(wayback_first=True)
    paywall_group = parser.add_mutually_exclusive_group()
    paywall_group.add_argument(
        "--llm-paywall-check",
        dest="llm_paywall_check",
        action="store_true",
        help="Use an OpenAI model as secondary paywall detector when heuristics are inconclusive (default).",
    )
    paywall_group.add_argument(
        "--no-llm-paywall-check",
        dest="llm_paywall_check",
        action="store_false",
        help="Disable LLM paywall checking and rely only on deterministic heuristics.",
    )
    parser.set_defaults(llm_paywall_check=True)
    nli_group = parser.add_mutually_exclusive_group()
    nli_group.add_argument(
        "--nli-stances",
        dest="nli_stance_check",
        action="store_true",
        help="Use OpenAI NLI to classify claim-citation stance (entailment/contradiction/neutral) (default).",
    )
    nli_group.add_argument(
        "--no-nli-stances",
        dest="nli_stance_check",
        action="store_false",
        help="Disable NLI stance classification and use heuristic stance labels only.",
    )
    parser.set_defaults(nli_stance_check=True)
    parser.add_argument(
        "--nli-batch-size",
        type=int,
        default=24,
        help="Claim-citation pairs per NLI request batch.",
    )
    parser.add_argument(
        "--max-wayback-urls",
        type=int,
        default=8,
        help="Maximum number of Wayback snapshot URLs to try per reference.",
    )
    parser.add_argument("--model-id", default=None, help="LangExtract model id override.")
    parser.add_argument(
        "--provider",
        choices=["auto", "openai"],
        default="openai",
        help="Model provider behavior. 'openai' enables OpenAI-specific settings.",
    )
    parser.add_argument(
        "--openai",
        action="store_true",
        help="Shortcut for --provider openai.",
    )
    parser.add_argument(
        "--openai-api-key",
        default=None,
        help="OpenAI API key. Defaults to OPENAI_API_KEY env var.",
    )
    parser.add_argument(
        "--openai-reasoning-effort",
        default=DEFAULT_OPENAI_REASONING_EFFORT,
        help=(
            "OpenAI reasoning effort (for GPT-5 reasoning models). "
            "Typical values: low|medium|high|minimal."
        ),
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show LangExtract progress bars.",
    )
    parser.add_argument(
        "--show-absl-errors",
        action="store_true",
        help="Show noisy internal absl parser errors from LangExtract retries.",
    )
    parser.add_argument("--max-chars", type=int, default=120_000, help="Max chars read from document.")
    parser.add_argument(
        "--no-langextract",
        action="store_true",
        help="Skip LangExtract and use deterministic fallback matching.",
    )
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout in seconds.")
    parser.add_argument(
        "--include-document",
        action="store_true",
        help="Include full document text in JSON output (can be large).",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=1500,
        help="Chars shown in document_preview when --include-document is not set.",
    )
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVEL_CHOICES,
        default=DEFAULT_LOG_LEVEL,
        help="Log verbosity for CLI stderr output.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mitre-langextract-mini",
        description="Claim Flow V2 extractor and evaluator.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser(
        "extract",
        help="Grab one MITRE reference document (random or --reference-url) and run extraction.",
    )
    _add_extract_args(extract_parser)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate V2 claim assessments and associations against a gold payload.",
    )
    evaluate_parser.add_argument("--pred", required=True, help="Prediction JSON/JSONL file or directory.")
    evaluate_parser.add_argument("--gold", required=True, help="Gold JSON/JSONL file or directory.")
    evaluate_parser.add_argument("--format", choices=["json", "markdown"], default="json")
    evaluate_parser.add_argument("--span-iou-threshold", type=float, default=0.5)
    evaluate_parser.add_argument("--quote-similarity-threshold", type=float, default=0.8)
    evaluate_parser.add_argument(
        "--log-level",
        choices=LOG_LEVEL_CHOICES,
        default=DEFAULT_LOG_LEVEL,
        help="Log verbosity for CLI stderr output.",
    )
    return parser


def _run_extract_command(args: argparse.Namespace) -> int:
    configure_logging(args.log_level)
    provider = "openai" if args.openai else args.provider
    LOGGER.info("CLI start: provider=%s log_level=%s", provider, args.log_level)
    try:
        result = run_random_reference_langextract(
            bundle_url=args.bundle_url,
            cache_path=args.cache_path,
            output_dir=args.output_dir,
            reference_url=args.reference_url,
            prefer_wayback=args.wayback_first,
            llm_paywall_check=args.llm_paywall_check,
            max_wayback_urls=max(1, int(args.max_wayback_urls)),
            model_id=args.model_id,
            provider=provider,
            openai_api_key=args.openai_api_key,
            openai_reasoning_effort=args.openai_reasoning_effort,
            nli_stance_check=args.nli_stance_check,
            nli_batch_size=max(1, int(args.nli_batch_size)),
            show_progress=args.show_progress,
            quiet_absl=not args.show_absl_errors,
            use_langextract=not args.no_langextract,
            max_chars=args.max_chars,
            timeout=args.timeout,
        )
        payload = to_cli_payload(
            result,
            include_document=args.include_document,
            preview_chars=args.preview_chars,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        LOGGER.info("CLI success")
        return 0
    except ReferenceFetchError as exc:
        LOGGER.error("CLI failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        print(f"Failure reason: {exc.failure_reason}", file=sys.stderr)
        if exc.attempts:
            preview = exc.attempts[:5]
            print(f"Fetch attempts (first {len(preview)}):", file=sys.stderr)
            for attempt in preview:
                status = attempt.get("status")
                reason = attempt.get("failure_reason") or attempt.get("reason") or ""
                print(
                    f"- {attempt.get('source')} {attempt.get('url')} -> {status}"
                    + (f" ({reason})" if reason else ""),
                    file=sys.stderr,
                )
        return 2
    except RuntimeError as exc:
        LOGGER.error("CLI failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        if "OPENAI_API_KEY" in str(exc):
            print(
                "Hint: export OPENAI_API_KEY or pass --openai-api-key, "
                "or run with --no-langextract.",
                file=sys.stderr,
            )
        return 2


def _run_evaluate_command(args: argparse.Namespace) -> int:
    configure_logging(args.log_level)
    try:
        result = eval_v2.evaluate_prediction_files(
            pred_path=args.pred,
            gold_path=args.gold,
            span_iou_threshold=float(args.span_iou_threshold),
            quote_similarity_threshold=float(args.quote_similarity_threshold),
        )
        if args.format == "markdown":
            print(eval_v2.to_markdown_report(result))
        else:
            print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except RuntimeError as exc:
        LOGGER.error("Evaluation failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "extract":
        return _run_extract_command(args)
    if args.command == "evaluate":
        return _run_evaluate_command(args)
    print("ERROR: unknown command", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
