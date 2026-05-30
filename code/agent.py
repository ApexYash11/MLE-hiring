from __future__ import annotations

import json
import logging
import os
import pathlib
import threading
import time

import config as _config
from openai import APIError, OpenAI

from config import (
    MAX_CONVERSATION_TOKENS,
    MAX_LLM_RETRIES,
    MAX_RETRIEVAL_TOKENS,
    RETRY_BACKOFF_SECONDS,
    SAFE_ESCALATION_CONFIDENCE,
    MODEL,
    GROQ_BASE_URL,
    GEMINI_BASE_URL,
    OPENROUTER_BASE_URL,
    DEEPSEEK_BASE_URL,
    FALLBACK_MODEL_CHAIN,
    TEMPERATURE,
    MAX_TOKENS,
    PER_MODEL_TIMEOUT,
)
from prompts import CONVERSATION_SUMMARY_PROMPT, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


class AgentRunner:
    def __init__(self):
        groq_key = os.environ.get("GROQ_API_KEY") or os.environ.get("groq_api_key")
        self._groq_client = (
            OpenAI(
                base_url=GROQ_BASE_URL,
                api_key=groq_key,
                max_retries=0,
            )
            if groq_key
            else None
        )

        gemini_key = os.environ.get("GEMINI_API_KEY")
        self._gemini_client = (
            OpenAI(
                base_url=GEMINI_BASE_URL,
                api_key=gemini_key,
                max_retries=0,
            )
            if gemini_key
            else None
        )

        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        self._openrouter_client = (
            OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=openrouter_key,
                max_retries=0,
            )
            if openrouter_key
            else None
        )

        deepseek_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get(
            "Deepseek_api_key"
        )
        self._deepseek_client = (
            OpenAI(
                base_url=DEEPSEEK_BASE_URL,
                api_key=deepseek_key,
                max_retries=0,
            )
            if deepseek_key
            else None
        )

        if (
            not self._groq_client
            and not self._gemini_client
            and not self._openrouter_client
            and not self._deepseek_client
        ):
            raise RuntimeError("No API keys found.")

        self._extra_headers = {
            "HTTP-Referer": "https://github.com/Rithvik1709/MLE-hiring",
            "X-Title": "MLE Hiring Challenge",
        }

        tools_schema_path = getattr(
            _config,
            "TOOLS_SCHEMA",
            str(
                pathlib.Path(__file__).resolve().parent.parent
                / "data"
                / "api_specs"
                / "internal_tools.json"
            ),
        )
        with open(tools_schema_path, encoding="utf-8") as f:
            self._tool_schema = json.load(f)

        logger.info(
            "AgentRunner initialized (Groq=%s, Gemini=%s, OpenRouter=%s)",
            groq_key is not None,
            gemini_key is not None,
            openrouter_key is not None,
        )

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
                conversation_text = (
                    f"Earlier in this conversation: {summary}\n\n"
                    + "\n".join(recent_lines)
                )
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
            f"[Doc {index}] {doc.get('text', '')}"
            for index, doc in enumerate(kept_docs, start=1)
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

        clients = [
            ("Groq", getattr(self, "_groq_client", None)),
            ("Gemini", getattr(self, "_gemini_client", None)),
        ]
        for name, client in clients:
            if client is None:
                continue
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    temperature=0,
                    max_tokens=150,
                    messages=[{"role": "user", "content": prompt}],
                )
                content = response.choices[0].message.content or ""
                return content.strip() or " ".join(conversation_text.split()[:200])
            except Exception as exc:
                logger.warning("%s summarization failed: %s", name, exc)

        return " ".join(conversation_text.split()[:200])

    def _call_with_timeout(
        self, client: OpenAI, model: str, prompt: str, timeout: int
    ) -> str:
        result: list[str | None] = [None]
        error: list[Exception | None] = [None]

        def _do_call(with_json_mode: bool):
            kwargs = dict(
                model=model,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                extra_headers=self._extra_headers,
            )
            if with_json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            return client.chat.completions.create(**kwargs)

        def target():
            try:
                resp = _do_call(with_json_mode=True)
                if resp.choices:
                    result[0] = resp.choices[0].message.content or ""
            except APIError as e:
                code = getattr(e, "status_code", 0)
                if code in (400, 422) and "response_format" in str(e).lower():
                    try:
                        resp = _do_call(with_json_mode=False)
                        if resp.choices:
                            result[0] = resp.choices[0].message.content or ""
                    except Exception as e2:
                        error[0] = e2
                else:
                    error[0] = e
            except Exception as e:
                error[0] = e

        t = threading.Thread(target=target, daemon=True)
        t.start()
        t.join(timeout=timeout)

        if t.is_alive():
            raise TimeoutError(f"Model {model} timed out after {timeout}s")
        if error[0]:
            raise error[0]
        return result[0]  # type: ignore[return-value]

    def _extract_retry_delay(self, exc: Exception) -> float | None:
        err_str = str(exc).lower()
        import re

        for pattern in [
            r"retry[_\s]?after[_\s]?seconds[_\s]?[:=]\s*(\d+)",
            r"retry[_\s]?delay[_\s]?:\s*(\d+)s",
            r"please retry in\s*(\d+(?:\.\d+)?)s",
        ]:
            m = re.search(pattern, err_str)
            if m:
                return float(m.group(1))
        return None

    def _call_llm(self, prompt: str) -> str:
        if self._deepseek_client:
            try:
                result = self._call_with_timeout(
                    self._deepseek_client, MODEL, prompt, PER_MODEL_TIMEOUT
                )
                if result:
                    logger.info("LLM success (DeepSeek): %s", MODEL)
                    return result
            except Exception as e:
                delay = self._extract_retry_delay(e)
                if delay:
                    logger.warning("DeepSeek rate-limited, waiting %.0fs: %s", delay, e)
                    time.sleep(delay)
                else:
                    logger.warning("DeepSeek %s failed: %s", MODEL, e)

        if self._groq_client:
            try:
                result = self._call_with_timeout(
                    self._groq_client, MODEL, prompt, PER_MODEL_TIMEOUT
                )
                if result:
                    logger.info("LLM success (Groq): %s", MODEL)
                    return result
            except Exception as e:
                delay = self._extract_retry_delay(e)
                if delay:
                    logger.warning("Groq rate-limited, waiting %.0fs: %s", delay, e)
                    time.sleep(delay)
                else:
                    logger.warning("Groq %s failed: %s", MODEL, e)

        if self._gemini_client:
            try:
                result = self._call_with_timeout(
                    self._gemini_client, MODEL, prompt, PER_MODEL_TIMEOUT
                )
                if result:
                    logger.info("LLM success (Gemini): %s", MODEL)
                    return result
            except Exception as e:
                delay = self._extract_retry_delay(e)
                if delay:
                    logger.warning("Gemini rate-limited, waiting %.0fs: %s", delay, e)
                    time.sleep(delay)
                else:
                    logger.warning("Gemini %s failed: %s", MODEL, e)

        if self._openrouter_client:
            for model in FALLBACK_MODEL_CHAIN:
                try:
                    result = self._call_with_timeout(
                        self._openrouter_client, model, prompt, PER_MODEL_TIMEOUT
                    )
                    if result:
                        logger.info("LLM success via fallback: %s", model)
                        return result
                except Exception as e:
                    delay = self._extract_retry_delay(e)
                    if delay:
                        logger.warning(
                            "OpenRouter rate-limited, waiting %.0fs: %s", delay, e
                        )
                        time.sleep(delay)
                    else:
                        logger.warning("Fallback %s failed: %s", model, e)
                    continue

        raise RuntimeError("All providers and fallback models exhausted")

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
                raw_output = self._call_llm(user_prompt)
                break
            except Exception as exc:
                last_error = exc
                logger.warning("LLM attempt %d failed: %s", attempt + 1, exc)
                if attempt < MAX_LLM_RETRIES:
                    delay = self._extract_retry_delay(exc)
                    if delay is None:
                        delay = RETRY_BACKOFF_SECONDS * (2**attempt)
                    logger.info("Retrying in %.0fs...", delay)
                    time.sleep(delay)

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
    if "GEMINI_API_KEY" not in os.environ:
        print("SKIP: GEMINI_API_KEY not set")
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
