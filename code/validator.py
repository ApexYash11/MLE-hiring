from __future__ import annotations

import json
import logging
import re
from typing import Any, List, Literal

from json_repair import repair_json
from pydantic import BaseModel, Field, field_validator

from config import SAFE_ESCALATION_CONFIDENCE, CONTRADICTION_MAX_CONFIDENCE

try:
    from config import TOOLS_SCHEMA
except Exception:
    from pathlib import Path

    TOOLS_SCHEMA = str(Path(__file__).resolve().parent.parent / "data" / "api_specs" / "internal_tools.json")

logger = logging.getLogger(__name__)


class TicketOutput(BaseModel):
    status: Literal["replied", "escalated"]
    product_area: str
    response: str
    justification: str
    request_type: Literal["product_issue", "feature_request", "bug", "invalid"]
    confidence_score: float = Field(ge=0.0, le=1.0)
    risk_level: Literal["low", "medium", "high", "critical"]
    actions_taken: List[Any] = []
    reasoning: str = ""

    @field_validator("response")
    @classmethod
    def response_not_empty(cls, v):
        if not v or not str(v).strip():
            raise ValueError("response cannot be empty")
        return str(v).strip()

    @field_validator("confidence_score")
    @classmethod
    def confidence_in_range(cls, v):
        return round(max(0.0, min(1.0, float(v))), 4)


class OutputValidator:
    def __init__(self):
        with open(TOOLS_SCHEMA, encoding="utf-8") as f:
            self._tool_schema = json.load(f)

        self._allowed_tools = set()
        for item in self._tool_schema:
            if isinstance(item, dict) and item.get("name"):
                self._allowed_tools.add(str(item["name"]))

        self._destructive_tools = {
            "issue_refund",
            "lock_account",
            "delete_data",
            "modify_account",
        }

        logger.info("OutputValidator initialized")

    def validate(self, raw_output: str | dict, ticket_id: str) -> dict:
        if isinstance(raw_output, dict):
            try:
                parsed = TicketOutput(**raw_output)
                return self._finalize(parsed)
            except Exception:
                pass

        raw_text = raw_output if isinstance(raw_output, str) else json.dumps(raw_output)

        try:
            data = json.loads(raw_text)
            parsed = TicketOutput(**data)
            return self._finalize(parsed)
        except Exception:
            pass

        try:
            repaired = repair_json(raw_text)
            data = json.loads(repaired)
            parsed = TicketOutput(**data)
            return self._finalize(parsed)
        except Exception:
            pass

        try:
            cleaned = raw_text
            cleaned = re.sub(r"```json\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"```\s*", "", cleaned)
            cleaned = cleaned.strip()
            data = json.loads(cleaned)
            parsed = TicketOutput(**data)
            return self._finalize(parsed)
        except Exception:
            pass

        try:
            match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if match:
                data = json.loads(match.group())
                parsed = TicketOutput(**data)
                return self._finalize(parsed)
        except Exception:
            pass

        logger.error("[%s] All validation attempts failed, returning safe escalation", ticket_id)
        return self._safe_escalation_output("Output validation failed after all recovery attempts")

    def _finalize(self, parsed: TicketOutput) -> dict:
        result = parsed.model_dump()

        validated_actions = []
        needs_verify = False

        for action in result.get("actions_taken", []):
            if not isinstance(action, dict):
                continue
            tool_name = action.get("tool") or action.get("name") or ""
            if tool_name not in self._allowed_tools:
                logger.warning("Hallucinated tool rejected: %s", tool_name)
                continue
            if tool_name in self._destructive_tools:
                needs_verify = True
            validated_actions.append(action)

        if needs_verify:
            has_verify = any((a.get("tool") or a.get("name")) == "verify_identity" for a in validated_actions)
            if not has_verify:
                validated_actions.insert(0, {"tool": "verify_identity", "args": {}})

        result["actions_taken"] = validated_actions
        result.pop("reasoning", None)
        return result

    def _safe_escalation_output(self, reason: str) -> dict:
        return {
            "status": "escalated",
            "product_area": "unknown",
            "response": "This request has been routed to our team for review. A human agent will follow up shortly.",
            "justification": reason,
            "request_type": "invalid",
            "confidence_score": SAFE_ESCALATION_CONFIDENCE,
            "risk_level": "high",
            "actions_taken": [],
        }


if __name__ == "__main__":
    v = OutputValidator()

    tests = [
        ('{"status":"replied","product_area":"billing","response":"Here is your answer","justification":"clear","request_type":"product_issue","confidence_score":0.85,"risk_level":"low","actions_taken":[],"reasoning":"x"}', "replied"),
        ('{"status":"replied","product_area":"billing","response":"Here is your answer","justification":"clear","request_type":"product_issue","confidence_score":0.85,"risk_level":"low","actions_taken":[', "replied"),
        ('```json\n{"status":"escalated","product_area":"security","response":"Escalated","justification":"risk","request_type":"invalid","confidence_score":0.92,"risk_level":"high","actions_taken":[]}\n```', "escalated"),
        ("not json at all %%%", "escalated"),
        ('{"status":"replied","product_area":"billing","response":"Done","justification":"ok","request_type":"product_issue","confidence_score":0.8,"risk_level":"low","actions_taken":[{"tool":"fake_tool","args":{}}]}', "replied"),
        ('{"status":"replied","product_area":"billing","response":"Refund issued","justification":"eligible","request_type":"product_issue","confidence_score":0.8,"risk_level":"medium","actions_taken":[{"tool":"issue_refund","args":{}}]}', "replied"),
    ]

    passed = 0
    for raw, expected_status in tests:
        result = v.validate(raw, "test-001")
        actual = result.get("status")
        ok = actual == expected_status
        passed += int(ok)
        if not ok:
            print(f"FAIL | expected={expected_status} got={actual}")
            print(f"      input={raw[:80]}")

    refund_result = v.validate(tests[5][0], "test-refund")
    actions = refund_result.get("actions_taken", [])
    prepend_ok = bool(actions) and (actions[0].get("tool") or actions[0].get("name")) == "verify_identity"
    if prepend_ok:
        passed += 1
        print("PASS | verify_identity prepend for destructive tool")
    else:
        print(f"FAIL | verify_identity not prepended | actions={actions}")

    print(f"\n{passed}/{len(tests)+1} tests passed")
    if passed == len(tests) + 1:
        print("validator.py OK")
    else:
        print("Fix failures before proceeding to Step 8.")