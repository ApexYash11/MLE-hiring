from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langdetect import detect as detect_language
from langdetect.lang_detect_exception import LangDetectException

import config as _config
import agent as agent_module
from pii import PIIDetector
from retriever import RetrievalEngine
from safety import SafetyEngine
from validator import OutputValidator

log_dir = Path.home() / "mle_hiring"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "log.txt", mode="a"),
        logging.StreamHandler(sys.stderr),
    ],
)
logger = logging.getLogger("main")

_BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = getattr(_config, "DATA_DIR", _BASE_DIR / "data")
INPUT_CSV = getattr(_config, "INPUT_CSV", _BASE_DIR / "support_tickets" / "support_tickets.csv")
OUTPUT_CSV = getattr(_config, "OUTPUT_CSV", _BASE_DIR / "support_tickets" / "output.csv")
RETRIEVAL_TOP_K = getattr(_config, "RETRIEVAL_TOP_K", 5)
SAFE_ESCALATION_CONFIDENCE = getattr(_config, "SAFE_ESCALATION_CONFIDENCE", 0.92)

OUTPUT_COLUMNS = [
    "issue",
    "subject",
    "company",
    "response",
    "product_area",
    "status",
    "request_type",
    "justification",
    "confidence_score",
    "source_documents",
    "risk_level",
    "pii_detected",
    "language",
    "actions_taken",
]


def classify_product(ticket_row: dict) -> str:
    company = str(ticket_row.get("company", "")).lower().strip()
    subject = str(ticket_row.get("subject", "")).lower()
    conversation = str(ticket_row.get("conversation", "")).lower()
    combined = subject + " " + conversation

    if "devplatform" in company or "devplatform" in combined:
        return "devplatform"
    elif "claude" in company or "anthropic" in combined:
        return "claude"
    elif "visa" in company or "visa" in combined:
        return "visa"

    if any(w in combined for w in ["api", "sdk", "webhook", "endpoint", "token", "pipeline"]):
        return "devplatform"
    if any(w in combined for w in ["refund", "charge", "transaction", "card", "payment", "dispute", "billing"]):
        return "visa"
    if any(w in combined for w in ["model", "prompt", "completion", "claude", "context", "inference", "rate limit"]):
        return "claude"

    return "general"


def detect_lang(text: str) -> str:
    try:
        return detect_language(str(text)[:500])
    except LangDetectException:
        return "en"
    except Exception:
        return "en"


def build_safe_row(ticket_row: dict, reason: str, pii_flag: bool, language: str) -> dict:
    return {
        "issue": ticket_row.get("issue", ""),
        "subject": ticket_row.get("subject", ""),
        "company": ticket_row.get("company", ""),
        "response": "This request has been routed to our team for review. A human agent will follow up shortly.",
        "product_area": "unknown",
        "status": "escalated",
        "request_type": "invalid",
        "justification": f"Pipeline error: {reason}",
        "confidence_score": SAFE_ESCALATION_CONFIDENCE,
        "source_documents": "",
        "risk_level": "high",
        "pii_detected": str(pii_flag).lower(),
        "language": language,
        "actions_taken": "[]",
    }


def process_ticket(
    row: dict,
    ticket_idx: int,
    pii_detector: PIIDetector,
    safety_engine: SafetyEngine,
    retrieval_engine: RetrievalEngine,
    agent_runner: AgentRunner,
    validator: OutputValidator,
) -> dict:
    ticket_id = str(row.get("ticket_id", f"row-{ticket_idx}"))

    raw_text = " ".join([
        str(row.get("subject", "")),
        str(row.get("conversation", "")),
        str(row.get("message", "")),
    ]).strip()

    language = detect_lang(raw_text)

    pii_flag = pii_detector.detect(raw_text)
    clean_text = pii_detector.redact(raw_text)
    clean_row = dict(row)
    clean_row["conversation"] = pii_detector.redact(str(row.get("conversation", "")))
    clean_row["subject"] = pii_detector.redact(str(row.get("subject", "")))

    is_adversarial, safety_reason = safety_engine.scan(raw_text)
    if is_adversarial:
        logger.warning("[%s] Injection detected: %s", ticket_id, safety_reason)
        return {
            **build_safe_row(row, safety_reason, pii_flag, language),
            "justification": safety_reason,
            "risk_level": "critical",
            "confidence_score": 0.92,
            "product_area": "security",
            "request_type": "invalid",
        }

    product = classify_product(row)

    try:
        docs = retrieval_engine.retrieve(query=clean_text, product=product, top_k=RETRIEVAL_TOP_K)
        contradiction_flag = retrieval_engine.has_contradiction(docs)
    except Exception as exc:
        logger.error("[%s] Retrieval failed: %s", ticket_id, exc)
        docs = []
        contradiction_flag = False

    try:
        raw_result = agent_runner.run(
            ticket_data=clean_row,
            retrieved_docs=docs,
            pii_flag=pii_flag,
            language=language,
            contradiction_flag=contradiction_flag,
        )
    except Exception as exc:
        logger.error("[%s] Agent failed: %s", ticket_id, exc)
        return build_safe_row(row, str(exc), pii_flag, language)

    raw_str = raw_result.get("_raw", raw_result)
    validated = validator.validate(raw_str, ticket_id)

    source_docs = "|".join(d["file_path"] for d in docs if d.get("file_path", "").strip())
    validated["source_documents"] = source_docs
    validated["pii_detected"] = str(pii_flag).lower()
    validated["language"] = language
    validated["actions_taken"] = json.dumps(validated.get("actions_taken", []), ensure_ascii=False)

    result = {
        "issue": row.get("issue", ""),
        "subject": row.get("subject", ""),
        "company": row.get("company", ""),
    }
    result.update({col: validated.get(col, "") for col in OUTPUT_COLUMNS if col not in result})
    return result


def main():
    load_dotenv()

    agent_module.MAX_LLM_RETRIES = 0
    agent_module.RETRY_BACKOFF_SECONDS = 0
    agent_module.MAX_TOKENS = min(getattr(agent_module, "MAX_TOKENS", 1000), 256)

    parser = argparse.ArgumentParser(description="Support triage agent pipeline")
    parser.add_argument("--input", default=str(INPUT_CSV), help="Path to input CSV")
    parser.add_argument("--output", default=str(OUTPUT_CSV), help="Path to output CSV")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    logger.info("%s", "=" * 60)
    logger.info("Support Triage Pipeline starting")
    logger.info("Input:  %s", input_path)
    logger.info("Output: %s", output_path)
    logger.info("%s", "=" * 60)

    logger.info("Loading modules...")
    pii_detector = PIIDetector()
    safety_engine = SafetyEngine()
    retrieval_engine = RetrievalEngine(str(DATA_DIR))
    agent_runner = agent_module.AgentRunner()
    validator = OutputValidator()
    logger.info("All modules loaded")

    df = pd.read_csv(input_path)
    total = len(df)
    logger.info("Loaded %d tickets", total)

    results = []
    start_time = time.time()

    for idx, row in df.iterrows():
        ticket_id = str(row.get("ticket_id", f"row-{idx}"))
        print(f"Processing ticket {idx + 1}/{total} [{ticket_id}]...", file=sys.stderr)
        try:
            result = process_ticket(
                row.to_dict(),
                idx,
                pii_detector,
                safety_engine,
                retrieval_engine,
                agent_runner,
                validator,
            )
        except Exception as exc:
            logger.error("[%s] Unhandled exception: %s", ticket_id, exc)
            result = build_safe_row(row.to_dict(), str(exc), False, "en")
        results.append(result)

    output_df = pd.DataFrame(results, columns=OUTPUT_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    elapsed = time.time() - start_time
    logger.info("Done. %d tickets in %.1fs (%.2fs/ticket)", total, elapsed, (elapsed / total) if total else 0.0)
    logger.info("Output written to %s", output_path)
    print(f"\nDone. {total} tickets processed in {elapsed:.1f}s", file=sys.stderr)
    print(f"Output: {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()