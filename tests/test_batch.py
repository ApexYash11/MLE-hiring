"""Comprehensive batch test suite — entry point.

Usage:
    python test_batch.py                  # Full run (main + sample CSV)
    python test_batch.py --quick          # 10 tickets from main + sample
    python test_batch.py --sample-only    # Sample CSV only
    python test_batch.py --input FILE     # Custom input CSV
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Suppress noisy loggers before loading modules
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from agent import AgentRunner
from pii import PIIDetector
from retriever import RetrievalEngine
from safety import SafetyEngine
from validator import OutputValidator

from tests.comparator import SampleComparator
from tests.config import INPUT_CSV, SAMPLE_CSV, DATA_DIR
from tests.diagnostics import DiagnosticEngine
from tests.reporter import print_report, save_results
from tests.runner import load_csv, run_batch

# Import model name for display
try:
    from config import MODEL as _MODEL
except ImportError:
    _MODEL = "unknown"


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive support triage test suite"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Sample CSV + 10 from main CSV"
    )
    parser.add_argument("--sample-only", action="store_true", help="Sample CSV only")
    parser.add_argument("--input", default=str(INPUT_CSV), help="Path to input CSV")
    args = parser.parse_args()

    print("=" * 60)
    print("  SUPPORT TRIAGE — TEST SUITE")
    print("  Model:", _MODEL)
    print("=" * 60)

    # Initialize pipeline modules once
    print("\nLoading modules...")
    sys.stdout.flush()
    pii = PIIDetector()
    print("  PIIDetector OK")
    sys.stdout.flush()
    safety = SafetyEngine()
    print("  SafetyEngine OK")
    sys.stdout.flush()
    retriever = RetrievalEngine(str(DATA_DIR))
    print("  RetrievalEngine OK")
    sys.stdout.flush()
    agent = AgentRunner()
    print("  AgentRunner OK")
    sys.stdout.flush()
    validator = OutputValidator()
    print("  Validator OK")
    sys.stdout.flush()

    from config import RETRIEVAL_TOP_K

    top_k = RETRIEVAL_TOP_K

    all_telemetries: list = []
    all_results: list[dict] = []
    comparison = None
    struct_errors: list[str] = []

    # ── Main CSV batch ──
    if not args.sample_only:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Input not found: {input_path}")
            sys.exit(1)

        n = 10 if args.quick else None
        tickets = load_csv(input_path, n)
        tele, results = run_batch(
            tickets,
            f"Main CSV: {input_path.name} ({'first 10' if args.quick else 'all'})",
            pii,
            safety,
            retriever,
            agent,
            validator,
            top_k,
            _MODEL,
        )
        all_telemetries.extend(tele)
        all_results.extend(results)

    # ── Sample CSV batch ──
    sample_tickets = load_csv(SAMPLE_CSV)
    sample_tele, sample_results = run_batch(
        sample_tickets,
        "Sample CSV",
        pii,
        safety,
        retriever,
        agent,
        validator,
        top_k,
        _MODEL,
    )
    all_telemetries.extend(sample_tele)
    all_results.extend(sample_results)

    # ── Compare with expected values ──
    comparator = SampleComparator(SAMPLE_CSV)
    comparison = comparator.compare(sample_results, sample_tele)

    # ── Diagnostics ──
    diag = DiagnosticEngine()
    diagnostics = diag.analyze(all_telemetries, comparison, len(struct_errors))

    # ── Report ──
    label = f"Full ({'quick' if args.quick else 'complete'})"
    print_report(
        all_telemetries,
        all_results,
        comparison,
        struct_errors,
        diagnostics,
        label,
        _MODEL,
    )

    # ── Save ──
    save_results(all_telemetries, all_results, comparison, diagnostics, label, _MODEL)

    # Exit code
    err_count = sum(1 for t in all_telemetries if t.status_code == "ERROR")
    ok_count = sum(1 for t in all_telemetries if t.status_code == "OK")
    if err_count > ok_count:
        print("Too many errors — exiting with code 1")
        sys.exit(1)


if __name__ == "__main__":
    main()
