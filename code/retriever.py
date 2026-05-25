from __future__ import annotations

import logging
import os
import pathlib

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

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

            self._embed_model = self._load_embed_model()
            texts = [chunk["text"] for chunk in self._chunks]
            embeddings = self._embed_model.encode(
                texts,
                batch_size=64,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            embeddings = np.asarray(embeddings)
            dim = embeddings.shape[1]
            self._faiss_index = faiss.IndexFlatIP(dim)
            self._faiss_index.add(embeddings.astype(np.float32))
            self._embeddings = embeddings
        else:
            self._bm25 = None
            self._embed_model = None
            self._faiss_index = None
            self._embeddings = np.empty((0, 0), dtype=np.float32)

        self._reranker = self._load_reranker()

        logger.info(
            "RetrievalEngine ready: %d chunks from %d files",
            len(self._chunks),
            len(file_paths_seen),
        )

    def retrieve(self, query: str, product: str = None, top_k: int = 5) -> list[dict]:
        if not self._chunks or self._bm25 is None or self._embed_model is None or self._faiss_index is None:
            return []

        tokenized_query = query.lower().split()
        bm25_scores = self._bm25.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:RETRIEVAL_CANDIDATE_K]
        bm25_results = {
            int(idx): {"rank": rank, "score": float(bm25_scores[idx])}
            for rank, idx in enumerate(bm25_top_indices)
        }

        query_embedding = self._embed_model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, indices = self._faiss_index.search(query_embedding, RETRIEVAL_CANDIDATE_K)
        semantic_results = {
            int(idx): {"rank": rank, "score": float(scores[0][rank])}
            for rank, idx in enumerate(indices[0])
            if idx >= 0
        }

        all_indices = set(bm25_results.keys()) | set(semantic_results.keys())
        rrf_scores: dict[int, float] = {}
        k = 60
        for idx in all_indices:
            score = 0.0
            if idx in bm25_results:
                score += 1.0 / (k + bm25_results[idx]["rank"] + 1)
            if idx in semantic_results:
                score += 1.0 / (k + semantic_results[idx]["rank"] + 1)
            rrf_scores[idx] = score

        if product is not None:
            product_lower = product.lower()
            for idx in rrf_scores:
                if self._chunks[idx]["product"] == product_lower:
                    rrf_scores[idx] *= PRODUCT_BOOST_FACTOR

        top_indices = sorted(rrf_scores, key=lambda idx: rrf_scores[idx], reverse=True)[:RETRIEVAL_CANDIDATE_K]
        candidates = [self._chunks[i] for i in top_indices]

        if not candidates:
            return []

        pairs = [[query, candidate["text"]] for candidate in candidates]
        ce_scores = self._reranker.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(ce_scores, candidates), key=lambda item: item[0], reverse=True)[:top_k]

        results = []
        for ce_score, chunk in ranked:
            results.append(
                {
                    "text": chunk["text"],
                    "file_path": chunk["file_path"],
                    "product": chunk["product"],
                    "trust_level": chunk["trust_level"],
                    "score": float(ce_score),
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

    def _chunk_text(self, text: str, file_path: str, product: str, trust_level: int) -> list[dict]:
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

    def _load_reranker(self):
        token_keys = [
            "HF_TOKEN",
            "HUGGINGFACE_HUB_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
            "HF_API_TOKEN",
            "HF_HUB_TOKEN",
        ]
        saved_tokens = {key: os.environ.get(key) for key in token_keys}
        previous_disable_implicit_token = os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN")

        try:
            for key in token_keys:
                os.environ.pop(key, None)
            os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
            return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
        except Exception as exc:
            logger.warning("CrossEncoder unavailable, using local fallback reranker: %s", exc)

            class _FallbackReranker:
                def predict(self, pairs, show_progress_bar: bool = False):
                    scores = []
                    for query, text in pairs:
                        query_terms = set(query.lower().split())
                        text_terms = set(text.lower().split())
                        overlap = len(query_terms & text_terms)
                        length_bonus = min(len(text_terms) / 1000.0, 1.0)
                        scores.append(float(overlap + length_bonus))
                    return np.asarray(scores, dtype=np.float32)

            return _FallbackReranker()
        finally:
            for key, value in saved_tokens.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

            if previous_disable_implicit_token is None:
                os.environ.pop("HF_HUB_DISABLE_IMPLICIT_TOKEN", None)
            else:
                os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = previous_disable_implicit_token

    def _load_embed_model(self):
        token_keys = [
            "HF_TOKEN",
            "HUGGINGFACE_HUB_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
            "HF_API_TOKEN",
            "HF_HUB_TOKEN",
        ]
        saved_tokens = {key: os.environ.get(key) for key in token_keys}
        previous_disable_implicit_token = os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN")

        try:
            for key in token_keys:
                os.environ.pop(key, None)
            os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
            return SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as exc:
            logger.warning("SentenceTransformer unavailable, using local fallback embedder: %s", exc)

            class _FallbackEmbedder:
                dim = 384

                def encode(
                    self,
                    texts,
                    batch_size: int = 64,
                    show_progress_bar: bool = False,
                    normalize_embeddings: bool = True,
                ):
                    vectors = []
                    for text in texts:
                        vector = np.zeros(self.dim, dtype=np.float32)
                        for token in text.lower().split():
                            index = abs(hash(token)) % self.dim
                            vector[index] += 1.0
                        if normalize_embeddings:
                            norm = float(np.linalg.norm(vector))
                            if norm > 0:
                                vector /= norm
                        vectors.append(vector)
                    return np.vstack(vectors)

            return _FallbackEmbedder()
        finally:
            for key, value in saved_tokens.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

            if previous_disable_implicit_token is None:
                os.environ.pop("HF_HUB_DISABLE_IMPLICIT_TOKEN", None)
            else:
                os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = previous_disable_implicit_token


if __name__ == "__main__":
    engine = RetrievalEngine(str(pathlib.Path(__file__).resolve().parent.parent / "data"))
    results = engine.retrieve("how do I reset my password", product="claude", top_k=5)
    print(f"Retrieved {len(results)} chunks")
    for r in results:
        print(f"  score={r['score']:.3f} | {r['file_path']}")
        assert r["file_path"].startswith("/") or "data" in r["file_path"], \
            "file_path must be a real path from the index"
    print("retriever.py OK")