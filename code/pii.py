"""PII detection and redaction utilities for support tickets."""

from __future__ import annotations

import re


class PIIDetector:
    """Detects and redacts common personally identifiable information."""

    def __init__(self) -> None:
        self.credit_card_pattern = re.compile(r"\b(?:\d[ -]?){13,16}\b")
        self.card_phrase_pattern = re.compile(
            r"\bcard\s+(?:number\s+)?(?:is|:)?\s*(?P<number>(?:\d[ -]?){13,16})\b",
            re.IGNORECASE,
        )
        self.ssn_pattern = re.compile(r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b")
        self.email_pattern = re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        )
        self.phone_pattern = re.compile(
            r"""
            (?<!\w)
            (?:\+\d{1,3}[\s.-]?)?
            (?:
                \(\d{2,4}\)[\s.-]?|
                \d{2,4}[\s.-]?
            )?
            \d{3,4}[\s.-]?\d{4}
            (?!\w)
            """,
            re.VERBOSE,
        )
        self.address_pattern = re.compile(
            r"\b\d{1,6}\s+[A-Za-z0-9.' -]+?\s+"
            r"(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Lane|Ln|Dr|Drive)\b"
            r"(?:[.,]?\s*(?:Apt|Unit|Suite|Ste)\s*[A-Za-z0-9-]+)?",
            re.IGNORECASE,
        )
        self.passport_pattern = re.compile(r"\b[A-Z]{1,2}\d{6,9}\b")
        self.ip_pattern = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

    def detect(self, text: str) -> bool:
        """Return True when any supported PII type is present."""
        if not text:
            return False

        if self._contains_valid_credit_card(text):
            return True

        return any(
            pattern.search(text)
            for pattern in (
                self.ssn_pattern,
                self.email_pattern,
                self.phone_pattern,
                self.address_pattern,
                self.passport_pattern,
                self.ip_pattern,
            )
        )

    def redact(self, text: str) -> str:
        """Replace supported PII with stable placeholders."""
        if not text:
            return text

        redacted = self._redact_credit_cards(text)
        redacted = self.ssn_pattern.sub("[SSN REDACTED]", redacted)
        redacted = self.email_pattern.sub("[EMAIL REDACTED]", redacted)
        redacted = self.phone_pattern.sub("[PHONE REDACTED]", redacted)
        redacted = self.address_pattern.sub("[ADDRESS REDACTED]", redacted)
        redacted = self.passport_pattern.sub("[PASSPORT REDACTED]", redacted)
        redacted = self.ip_pattern.sub(self._redact_valid_ip, redacted)
        return redacted

    def _redact_credit_cards(self, text: str) -> str:
        def phrase_replacement(match: re.Match[str]) -> str:
            number = match.group("number")
            if self._is_luhn_valid(number):
                return "card ending in XXXX"
            return match.group(0)

        redacted = self.card_phrase_pattern.sub(phrase_replacement, text)

        def number_replacement(match: re.Match[str]) -> str:
            number = match.group(0)
            if self._is_luhn_valid(number):
                return "card ending in XXXX"
            return number

        return self.credit_card_pattern.sub(number_replacement, redacted)

    def _contains_valid_credit_card(self, text: str) -> bool:
        return any(
            self._is_luhn_valid(match.group(0))
            for match in self.credit_card_pattern.finditer(text)
        )

    def _is_luhn_valid(self, value: str) -> bool:
        digits = [int(char) for char in re.sub(r"\D", "", value)]
        if not 13 <= len(digits) <= 16:
            return False

        checksum = 0
        parity = len(digits) % 2
        for index, digit in enumerate(digits):
            if index % 2 == parity:
                digit *= 2
                if digit > 9:
                    digit -= 9
            checksum += digit
        return checksum % 10 == 0

    def _redact_valid_ip(self, match: re.Match[str]) -> str:
        value = match.group(0)
        octets = value.split(".")
        if all(0 <= int(octet) <= 255 for octet in octets):
            return "[IP REDACTED]"
        return value
