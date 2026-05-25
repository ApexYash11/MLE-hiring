from __future__ import annotations

import json
import logging
import os
import pathlib
import time

import config as _config
from openai import OpenAI

from config import (
    MAX_CONVERSATION_TOKENS,
    MAX_LLM_RETRIES,
    MAX_RETRIEVAL_TOKENS,
    MAX_TOKENS,
    MODEL,
    OPENROUTER_BASE_URL,
    RETRY_BACKOFF_SECONDS,
    SAFE_ESCALATION_CONFIDENCE,
    SEED,
    TEMPERATURE,
)
from prompts import CONVERSATION_SUMMARY_PROMPT, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


class AgentRunner:
    def __init__(self):
        self._client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        self._extra_headers = {
            "HTTP-Referer": "https://github.com/Rithvik1709/MLE-hiring",
            "X-Title": "MLE Hiring Challenge",
        }

        tools_schema_path = getattr(
            _config,
            "TOOLS_SCHEMA",
            str(pathlib.Path(__file__).resolve().parent.parent / "data" / "api_specs" / "internal_tools.json"),
        )
        with open(tools_schema_path, encoding="utf-8") as f:
            self._tool_schema = json.load(f)

        self._fallback_model = getattr(_config, "FALLBACK_MODEL", MODEL)
        logger.info("AgentRunner initialized")

    def _build_user_prompt(
        self,
        ticket_data: dict,
        retrieved_docs: list[dict],
        pii_flag: bool,
        language: str,
        contradiction_flag: bool,
    ) -> str:
        raw_convo = ticket_data.get("conversation", "") or ""
        conversation_text = raw_convo
        if len(raw_convo.split()) > MAX_CONVERSATION_TOKENS:
            lines = [line for line in raw_convo.splitlines() if line.strip()]
            recent_lines = lines[-6:] if len(lines) > 6 else lines
            earlier_lines = lines[:-6] if len(lines) > 6 else []
            summary_source = "\n".join(earlier_lines).strip()
            if summary_source:
                summary = self._summarize_conversation(summary_source)
                conversation_text = f"Earlier in this conversation: {summary}\n\n" + "\n".join(recent_lines)
            else:
                conversation_text = "\n".join(recent_lines)

        kept_docs: list[dict] = []
        total_words = 0
        for doc in retrieved_docs:
            doc_words = len((doc.get("text", "") or "").split())
            if kept_docs and total_words + doc_words > MAX_RETRIEVAL_TOKENS:
                continue
            kept_docs.append(doc)
            total_words += doc_words

        retrieved_docs_formatted = "\n\n".join(
            f"[Doc {index}] {doc.get('text', '')}" for index, doc in enumerate(kept_docs, start=1)
        )

        contradiction_note = ""
        if contradiction_flag:
            contradiction_note = (
                "Two or more retrieved documents disagree on a policy relevant to this ticket. "
                "Prefer the higher trust_level document. Note the conflict in your justification. "
                "Cap confidence_score at 0.60."
            )

        return USER_PROMPT_TEMPLATE.format(
            company=ticket_data.get("company", "unknown"),
            subject=ticket_data.get("subject", ""),
            conversation_text=conversation_text,
            language=language,
            pii_flag=str(pii_flag),
            retrieved_docs_formatted=retrieved_docs_formatted,
            contradiction_flag=str(contradiction_flag),
            contradiction_note=contradiction_note,
            tool_schema_json=json.dumps(self._tool_schema, indent=2),
        )

    def _summarize_conversation(self, conversation_text: str) -> str:
        prompt = CONVERSATION_SUMMARY_PROMPT.format(conversation_text=conversation_text)
        try:
            response = self._client.chat.completions.create(
                model=MODEL,
                temperature=0,
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}],
                extra_headers=self._extra_headers,
            )
            content = response.choices[0].message.content or ""
            return content.strip() or " ".join(conversation_text.split()[:200])
        except Exception as exc:
            logger.warning("Conversation summarization failed: %s", exc)
            return " ".join(conversation_text.split()[:200])

    def _call_llm(self, user_prompt: str, use_fallback: bool = False) -> str:
        model = self._fallback_model if use_fallback else MODEL
        response = self._client.chat.completions.create(
            model=model,
            temperature=TEMPERATURE,
            seed=SEED,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            extra_headers=self._extra_headers,
        )
        return response.choices[0].message.content or ""

    def run(
        self,
        ticket_data: dict,
        retrieved_docs: list[dict],
        pii_flag: bool,
        language: str,
        contradiction_flag: bool,
    ) -> dict:
        user_prompt = self._build_user_prompt(
            ticket_data,
            retrieved_docs,
            pii_flag,
            language,
            contradiction_flag,
        )

        raw_output = None
        last_error = None

        for attempt in range(MAX_LLM_RETRIES + 1):
            try:
                use_fallback = attempt == MAX_LLM_RETRIES
                raw_output = self._call_llm(user_prompt, use_fallback)
                break
            except Exception as exc:
                last_error = exc
                logger.warning("LLM attempt %d failed: %s", attempt + 1, exc)
                if attempt < MAX_LLM_RETRIES:
                    time.sleep(RETRY_BACKOFF_SECONDS * (2 ** attempt))

        if raw_output is None:
            logger.error("All LLM attempts failed: %s", last_error)
            return self._safe_escalation(
                f"LLM unavailable after {MAX_LLM_RETRIES + 1} attempts"
            )

        return {"_raw": raw_output}

    def _safe_escalation(self, reason: str) -> dict:
        return {
            "status": "escalated",
            "product_area": "unknown",
            "response": (
                "This request has been routed to our team for review."
                " A human agent will follow up shortly."
            ),
            "justification": reason,
            "request_type": "invalid",
            "confidence_score": SAFE_ESCALATION_CONFIDENCE,
            "risk_level": "high",
            "actions_taken": [],
            "reasoning": "",
        }


if __name__ == "__main__":
    if "OPENROUTER_API_KEY" not in os.environ:
        print("SKIP: OPENROUTER_API_KEY not set")
        raise SystemExit(0)

    runner = AgentRunner()
    fake_ticket = {
        "company": "Claude",
        "subject": "Cannot log in",
        "conversation": (
            "User: I cannot log in to my account.\n"
            "Agent: Have you tried resetting your password?"
        ),
    }
    fake_docs = [
        {
            "text": "To reset your password go to settings > security.",
            "file_path": "/data/claude/faq/account.md",
            "product": "claude",
            "trust_level": 2,
            "score": 0.91,
        }
    ]
    result = runner.run(
        ticket_data=fake_ticket,
        retrieved_docs=fake_docs,
        pii_flag=False,
        language="en",
        contradiction_flag=False,
    )
    assert "_raw" in result, "Expected raw output dict"
    print(f"Raw output: {result['_raw'][:200]}")
    print("agent.py OK")