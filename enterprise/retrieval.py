"""BM25 + Dense 混合检索，通过 RRF 融合，并支持 Redis 结果缓存。"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from collections import Counter

from enterprise.embedding import text_tokens

log = logging.getLogger(__name__)


def bm25(query: str, docs: list[dict]) -> list[tuple[str, float]]:
    """对候选文档计算 BM25 得分，按分数降序返回 (id, score)。"""
    if not docs:
        return []
    query_tokens = text_tokens(query)
    tokenized_docs = [text_tokens(doc["text"]) for doc in docs]
    lengths = [len(tokens) for tokens in tokenized_docs]
    avg_length = sum(lengths) / len(docs) if any(lengths) else 1.0
    document_frequency = Counter(
        token for tokens in tokenized_docs for token in set(tokens)
    )

    scores: list[tuple[str, float]] = []
    for doc, tokens, length in zip(docs, tokenized_docs, lengths):
        term_frequency = Counter(tokens)
        score = 0.0
        for term in set(query_tokens):
            frequency_in_docs = document_frequency.get(term, 0)
            if not frequency_in_docs:
                continue
            inverse_frequency = math.log(
                1 + (len(docs) - frequency_in_docs + 0.5) / (frequency_in_docs + 0.5)
            )
            frequency_in_doc = term_frequency.get(term, 0)
            if frequency_in_doc:
                denominator = frequency_in_doc + 1.2 * (
                    0.25 + 0.75 * length / avg_length
                )
                score += inverse_frequency * (frequency_in_doc * 2.2) / denominator
        scores.append((doc["id"], score))
    return sorted(scores, key=lambda item: item[1], reverse=True)


def rrf(*rankings: list[tuple[str, float]], k: int = 60) -> list[tuple[str, float]]:
    """倒数排名融合：按 1/(k+rank) 累加多路检索排名。"""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (key, score) in enumerate(ranking, 1):
            if score <= 0:
                continue
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


class Retriever:
    """检索器：支持 hybrid/dense/bm25，并以数据库版本号隔离缓存。"""

    def __init__(self, store, vector, embedder, settings, cache=None):
        self.store = store
        self.vector = vector
        self.embedder = embedder
        self.settings = settings
        self.cache = cache
        self._reranker = None

    def _cache_key(self, query: str, top_k: int, strategy: str) -> str:
        """把文档版本和检索参数纳入缓存键，避免文档变更后命中旧结果。"""
        revision = self.store.revision() if hasattr(self.store, "revision") else 0
        payload = {
            "revision": revision,
            "query": query,
            "top_k": top_k,
            "strategy": strategy,
            "rerank_enabled": bool(self.settings.rerank_enabled),
            "rerank_model": self.settings.rerank_model,
            "embedding_backend": self.settings.embedding_backend,
            "embedding_model": self.settings.embedding_model,
            "vector_backend": self.settings.vector_backend,
        }
        digest = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return f"rag:search:v1:{digest}"

    def search(
        self,
        query: str,
        top_k: int | None = None,
        strategy: str = "hybrid",
    ) -> list[dict]:
        """执行检索；缓存故障时自动回退到原始检索流程。"""
        top_k = self.settings.top_k if top_k is None else top_k
        query = query.strip()
        if not 1 <= top_k <= 20:
            raise ValueError("top_k 超出范围")
        if strategy not in {"hybrid", "dense", "bm25"}:
            raise ValueError("Unsupported retrieval strategy")
        if not query:
            return []

        cache_key = self._cache_key(query, top_k, strategy)
        if self.cache is not None and self.cache.enabled:
            cached = self.cache.get_json(cache_key)
            if isinstance(cached, list):
                log.debug("检索缓存命中 key=%s", cache_key)
                return cached

        docs = self.store.chunks()
        if not docs:
            results: list[dict] = []
            self._cache_results(cache_key, results)
            return results

        pool = min(max(top_k * 4, 20), len(docs))
        by_id = {doc["id"]: doc for doc in docs}
        sparse = bm25(query, docs)[:pool] if strategy in {"hybrid", "bm25"} else []
        dense = (
            self.vector.search(self.embedder.encode([query], query=True)[0], pool)
            if strategy in {"hybrid", "dense"}
            else []
        )

        if strategy == "dense":
            ranked = [(doc_id, 1 / (60 + rank)) for rank, (doc_id, _) in enumerate(dense, 1)]
        elif strategy == "bm25":
            ranked = [
                (doc_id, 1 / (60 + rank))
                for rank, (doc_id, score) in enumerate(sparse, 1)
                if score > 0
            ]
        else:
            ranked = rrf(
                [(doc_id, 1.0) for doc_id, _ in dense],
                [(doc_id, score) for doc_id, score in sparse if score > 0],
            )

        ranked = [(doc_id, score) for doc_id, score in ranked if doc_id in by_id]
        if self.settings.rerank_enabled and ranked:
            if self._reranker is None:
                from sentence_transformers import CrossEncoder

                self._reranker = CrossEncoder(
                    self.settings.rerank_model,
                    cache_folder=str(self.settings.embedding_cache),
                )
            pairs = [(query, by_id[doc_id]["text"]) for doc_id, _ in ranked]
            logits = self._reranker.predict(pairs)
            ranked = sorted(
                zip([doc_id for doc_id, _ in ranked], [float(value) for value in logits]),
                key=lambda item: item[1],
                reverse=True,
            )

        results = []
        for rank, (doc_id, score) in enumerate(ranked[:top_k], 1):
            item = by_id[doc_id]
            metadata = item["metadata"]
            results.append(
                {
                    "id": doc_id,
                    "doc_id": item["doc_id"],
                    "filename": metadata["filename"],
                    "chunk_index": item["idx"],
                    "text": item["text"],
                    "start_char": metadata["start_char"],
                    "end_char": metadata["end_char"],
                    "rank": rank,
                    "score": round(float(score), 6),
                }
            )

        self._cache_results(cache_key, results)
        return results

    def _cache_results(self, key: str, results: list[dict]) -> None:
        if self.cache is not None and self.cache.enabled:
            self.cache.set_json(key, results, self.settings.redis_cache_ttl)
