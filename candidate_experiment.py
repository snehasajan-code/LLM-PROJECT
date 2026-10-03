"""Opt-in one-versus-three candidate experiment; never used by the Streamlit UI."""

import argparse
import json
import os
from pathlib import Path
from runpy import run_path

from translation_pipeline import run_translation_request
from translation_request import TranslationRequest


EXPERIMENT_ENV = "TRANSLATION_CANDIDATE_EXPERIMENT"
SUPPORTED_COUNTS = (1, 3)


def experiment_enabled():
    return os.environ.get(EXPERIMENT_ENV) == "1"


def run_experiment_case(request, candidate_count, *, verifier=None, translator=None,
                        run_repair=False, repair_runner=None):
    """Run one request with timing metadata kept outside TranslationRequest."""
    if candidate_count not in SUPPORTED_COUNTS:
        raise ValueError("candidate_count must be 1 or 3")

    metrics = {}
    request = run_translation_request(
        request,
        verifier=verifier,
        translator=translator,
        candidate_count=candidate_count,
        experiment_metrics=metrics,
    )

    if run_repair:
        metrics["repair_triggered"] = False
        verification = request.verification_result or {}
        if verification.get("status") == "REVIEW":
            if repair_runner is None:
                from repair_engine import repair_translation
                repair_runner = repair_translation
            repair_runner(
                request.source_text, request.translation or "", verification,
                request.source_language, request.target_language,
                request.context, request.tone,
                selected_meaning=request.selected_meaning,
                semantic_constraint=request.semantic_constraint,
                request=request,
            )
            metrics["repair_triggered"] = True

    return {
        **metrics,
        "verification_status": (request.verification_result or {}).get("status"),
        "translation": request.translation,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare NLLB translation with one vs. three candidates."
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--text", help="Fixed source sentence for both runs")
    inputs.add_argument(
        "--matrix", action="store_true",
        help="Use the existing tests/test_translation_matrix.py cases without changing them",
    )
    parser.add_argument("--source", help="Source language, e.g. English (required with --text)")
    parser.add_argument("--target", help="Target language, e.g. Malayalam (required with --text)")
    parser.add_argument("--context", default="General")
    parser.add_argument("--tone", default="Neutral")
    parser.add_argument("--selected-meaning", default=None)
    parser.add_argument("--semantic-constraint", default=None)
    parser.add_argument(
        "--run-repair", action="store_true",
        help="Explicitly invoke existing repair on REVIEW outcomes and measure whether it ran",
    )
    args = parser.parse_args(argv)

    if not experiment_enabled():
        parser.error(f"Set {EXPERIMENT_ENV}=1 to enable this model-backed experiment.")

    from verifier import verify_translation
    if args.matrix:
        cases = run_path(str(Path(__file__).parent / "tests" / "test_translation_matrix.py"))["TEST_CASES"]
    else:
        if not args.source or not args.target:
            parser.error("--source and --target are required with --text")
        cases = [("custom", args.text, args.source, args.target, args.context)]

    for case_index, (case_id, text, source, target, context) in enumerate(cases):
        def make_request():
            request = TranslationRequest(
                source_text=text, source_language=source, target_language=target,
                context=context, tone=args.tone,
            )
            request.select_meaning(args.selected_meaning, args.semantic_constraint)
            return request

        # Warm the cached NLLB/Qwen paths once per input; warm-up metrics are
        # intentionally discarded so model startup does not skew either arm.
        run_experiment_case(
            make_request(), 1, verifier=verify_translation, run_repair=False
        )
        counts = SUPPORTED_COUNTS if case_index % 2 == 0 else tuple(reversed(SUPPORTED_COUNTS))
        for count in counts:
            result = run_experiment_case(
                make_request(), count, verifier=verify_translation,
                run_repair=args.run_repair,
            )
            result["case_id"] = case_id
            print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
