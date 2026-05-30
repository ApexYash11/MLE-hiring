from __future__ import annotations

import logging
import pathlib

import numpy as np
from rank_bm25 import BM25Okapi

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    PRODUCT_BOOST_FACTOR,
    RETRIEVAL_CANDIDATE_K,
)

logger = logging.getLogger(__name__)


class RetrievalEngine:
    def __init__(self, data_dir: str):
        self._data_dir = pathlib.Path(data_dir).resolve()
        self._chunks: list[dict] = []

        file_paths_seen: set[str] = set()
        for file_path in self._data_dir.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in {".md", ".txt"}:
                continue
            if "api_specs" in {part.lower() for part in file_path.parts}:
                continue

            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            product = self._detect_product(file_path)
            trust_level = self._detect_trust_level(file_path)
            chunks = self._chunk_text(text, str(file_path), product, trust_level)
            for chunk_index, chunk in enumerate(chunks):
                chunk["chunk_index"] = chunk_index
                self._chunks.append(chunk)
            file_paths_seen.add(str(file_path))

        if self._chunks:
            corpus = [chunk["text"].lower().split() for chunk in self._chunks]
            self._bm25 = BM25Okapi(corpus)
        else:
            self._bm25 = None

        logger.info(
            "RetrievalEngine ready: %d chunks from %d files (BM25 only)",
            len(self._chunks),
            len(file_paths_seen),
        )

    def retrieve(self, query: str, product: str = None, top_k: int = 5) -> list[dict]:
        if not self._chunks or self._bm25 is None:
            return []

        tokenized_query = query.lower().split()
        bm25_scores = self._bm25.get_scores(tokenized_query)
        top_indices = np.argsort(bm25_scores)[::-1][:RETRIEVAL_CANDIDATE_K]

        scored = []
        for idx in top_indices:
            score = float(bm25_scores[idx])
            chunk = self._chunks[idx]
            if product is not None and chunk["product"] == product.lower():
                score *= PRODUCT_BOOST_FACTOR
            scored.append((score, idx))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, idx in scored[:top_k]:
            chunk = self._chunks[idx]
            results.append(
                {
                    "text": chunk["text"],
                    "file_path": chunk["file_path"],
                    "product": chunk["product"],
                    "trust_level": chunk["trust_level"],
                    "score": score,
                }
            )
        return results

    def has_contradiction(self, results: list[dict]) -> bool:
        try:
            if len(results) < 2:
                return False
            first, second = results[0], results[1]
            if first.get("product") != second.get("product"):
                return False
            if first.get("trust_level") != second.get("trust_level"):
                return False
            text_a = first.get("text", "").lower()
            text_b = second.get("text", "").lower()
            pairs = [
                ("eligible", "not eligible"),
                ("can", "cannot"),
                ("allowed", "prohibited"),
                ("always", "never"),
                ("must", "must not"),
            ]
            for positive, negative in pairs:
                a_pos = positive in text_a
                a_neg = negative in text_a
                b_pos = positive in text_b
                b_neg = negative in text_b
                if (a_pos and b_neg) or (a_neg and b_pos):
                    return True
            return False
        except Exception:
            return False

    def _detect_product(self, file_path: pathlib.Path) -> str:
        parts = {part.lower() for part in file_path.parts}
        if "devplatform" in parts:
            return "devplatform"
        if "claude" in parts:
            return "claude"
        if "visa" in parts:
            return "visa"
        return "general"

    def _detect_trust_level(self, file_path: pathlib.Path) -> int:
        parts = {part.lower() for part in file_path.parts}
        if "policies" in parts:
            return 4
        if "docs" in parts:
            return 3
        if "faq" in parts:
            return 2
        return 1

    def _chunk_text(
        self, text: str, file_path: str, product: str, trust_level: int
    ) -> list[dict]:
        words = text.split()
        step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
        chunks = []
        for i in range(0, len(words), step):
            chunk = " ".join(words[i : i + CHUNK_SIZE])
            if len(chunk.strip()) < 20:
                continue
            chunks.append(
                {
                    "text": chunk,
                    "file_path": file_path,
                    "product": product,
                    "trust_level": trust_level,
                    "chunk_index": i // step,
                }
            )
        return chunks
