import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from pii import PIIDetector


class PIIDetectorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = PIIDetector()

    def test_detects_and_redacts_luhn_valid_credit_card(self) -> None:
        text = "my card is 4111 1111 1111 1111"

        self.assertTrue(self.detector.detect(text))
        self.assertEqual(self.detector.redact(text), "my card ending in XXXX")

    def test_rejects_luhn_invalid_credit_card_like_number(self) -> None:
        text = "reference 4111 1111 1111 1112"

        self.assertFalse(self.detector.detect(text))
        self.assertEqual(self.detector.redact(text), text)

    def test_detects_and_redacts_supported_pii_types(self) -> None:
        text = (
            "email test@example.com ssn 123-45-6789 phone +1 (415) 555-2671 "
            "at 123 Main St passport A1234567 ip 192.168.1.1"
        )

        self.assertTrue(self.detector.detect(text))
        self.assertEqual(
            self.detector.redact(text),
            "email [EMAIL REDACTED] ssn [SSN REDACTED] phone [PHONE REDACTED] "
            "at [ADDRESS REDACTED] passport [PASSPORT REDACTED] ip [IP REDACTED]",
        )

    def test_non_pii_text_is_not_flagged(self) -> None:
        self.assertFalse(self.detector.detect("not pii"))


if __name__ == "__main__":
    unittest.main()
