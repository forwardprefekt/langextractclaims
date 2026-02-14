"""Minimal MITRE reference -> LangExtract workflow package."""

from pathlib import Path
from typing import Any

__all__ = [
    "run_random_reference_langextract",
    "to_cli_payload",
    "evaluate_prediction_files",
    "evaluate_main",
    "main",
]


def run_random_reference_langextract(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from .workflow import run_random_reference_langextract as _impl

    return _impl(*args, **kwargs)


def to_cli_payload(
    result: dict[str, Any],
    *,
    include_document: bool,
    preview_chars: int,
) -> dict[str, Any]:
    from .workflow import to_cli_payload as _impl

    return _impl(
        result,
        include_document=include_document,
        preview_chars=preview_chars,
    )


def evaluate_prediction_files(
    *,
    pred_path: str | Path,
    gold_path: str | Path,
    span_iou_threshold: float = 0.5,
    quote_similarity_threshold: float = 0.8,
) -> dict[str, Any]:
    from .evaluate import evaluate_prediction_files as _impl

    return _impl(
        pred_path=pred_path,
        gold_path=gold_path,
        span_iou_threshold=span_iou_threshold,
        quote_similarity_threshold=quote_similarity_threshold,
    )


def evaluate_main(argv: list[str] | None = None) -> int:
    from .evaluate import main as _impl

    return _impl(argv)


def main(argv: list[str] | None = None) -> int:
    from .workflow import main as _impl

    return _impl(argv)
