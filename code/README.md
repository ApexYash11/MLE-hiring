# Code README

## What this agent does

This agent processes `support_tickets/support_tickets.csv` and writes a complete `support_tickets/output.csv` with all required columns.

The pipeline is deterministic where practical and is organized into the following stages:

1. Input loading and language detection.
2. PII detection and redaction.
3. Adversarial safety screening.
4. Hybrid retrieval over the shipped support corpus.
5. OpenRouter-based response generation.
6. Output validation and CSV writing.

## Requirements

- Python 3.10+
- `uv`
- Environment variables in `.env` or the shell for any model/API keys you use.

If you use OpenRouter, set `OPENROUTER_API_KEY` before running the agent.

## Install

From the repository root:

```bash
uv sync
```

If you only want to install the declared dependencies into the active environment, use:

```bash
uv pip install -r code/requirements.txt
```

## Run

Generate the submission CSV:

```bash
uv run python .\code\main.py --input .\support_tickets\support_tickets.csv --output .\support_tickets\output.csv
```

The script defaults to the same input and output paths, so the shorter form also works:

```bash
uv run python .\code\main.py
```

## Validate

Check the output structure after a run:

```bash
uv run python .\code\validate_output.py
```

Run the adversarial regression suite:

```bash
uv run python .\code\test_adversarial.py
```

## Output contract

The generated CSV must include all required columns in the exact order expected by the repository validator:

- issue
- subject
- company
- response
- product_area
- status
- request_type
- justification
- confidence_score
- source_documents
- risk_level
- pii_detected
- language
- actions_taken

## Notes

- The agent only uses the bundled `data/` corpus for grounded retrieval.
- Safety failures and model/runtime errors fall back to a safe escalation path instead of crashing the run.
- `output.csv` is intentionally generated at runtime and should not be hand-edited.
