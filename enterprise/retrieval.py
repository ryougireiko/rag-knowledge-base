"""稀疏 BM25 + 向量检索，通过 RRF 融合，并可选用 cross-encoder 重排。"""
from __future__ import annotations
import math
from collections import Counter
from functools import lru_cache
from enterprise.embedding import text_tokens


def bm25(query: str, docs: list[dict]) -> list[tuple[str,float]]:
    """对候选文档计算 BM25 得分，按分数降序返回 (id, score)。"""
    if not docs: return []
    qt = text_tokens(query)
    tokens = [text_tokens(d['text']) for d in docs]
    lengths = [len(t) for t in tokens]
    avg = sum(lengths)/len(docs) if any(lengths) else 1.0  # 平均文档长度
    df = Counter(t for ts in tokens for t in set(ts))  # 统计词项的文档频率
    scores = []
    for doc, toks, length in zip(docs, tokens, lengths):
        tf = Counter(toks)
        score = 0.0
        for term in set(qt):
            n = df.get(term, 0)
            if not n: continue
            idf = math.log(1+(len(docs)-n+0.5)/(n+0.5))  # 逆文档频率
            freq = tf.get(term,0)
            # BM25 核心公式，k1=2.2、b=0.25/0.75 为调节参数
            score += idf*(freq*2.2)/(freq+1.2*(0.25+0.75*length/avg)) if freq else 0
        scores.append((doc['id'],score))
    return sorted(scores,key=lambda x:x[1],reverse=True)


def rrf(*rankings: list[tuple[str,float]], k: int = 60) -> list[tuple[str,float]]:
    """倒数排名融合(RRF)：合并多路排序，分数为 1/(k+rank) 累加。"""
    scores: dict[str,float] = {}
    for ranking in rankings:
        for rank,(key,score) in enumerate(ranking,1):
            if score <= 0: continue
            scores[key] = scores.get(key,0)+1.0/(k+rank)
    return sorted(scores.items(), key=lambda x:x[1], reverse=True)


class Retriever:
    """检索器：支持 hybrid/dense/bm25 三种策略，可选重排。"""
    def __init__(self, store, vector, embedder, settings):
        self.store, self.vector, self.embedder, self.settings = store, vector, embedder, settings
        self._reranker = None  # 重排模型懒加载

    def search(self, query: str, top_k: int | None = None, strategy: str = 'hybrid') -> list[dict]:
        """执行检索：召回候选、融合排序、可选重排，返回 top_k 结果。"""
        top_k = top_k or self.settings.top_k
        if not 1 <= top_k <= 20: raise ValueError('top_k 超出范围')
        docs = self.store.chunks()
        if not docs or not query.strip(): return []
        pool = min(max(top_k*4,20), len(docs))  # 先扩大召回池，再精排
        if strategy not in {'hybrid','dense','bm25'}:
            raise ValueError('Unsupported retrieval strategy')
        by_id = {d['id']:d for d in docs}
        sparse = bm25(query, docs)[:pool] if strategy in {'hybrid','bm25'} else []
        dense = self.vector.search(self.embedder.encode([query],query=True)[0], pool) if strategy in {'hybrid','dense'} else []
        # 纯向量检索时余弦相似度可能为负，但仍可用于排序候选
        # For pure-vector search, cosine similarities may be < 0, yet still rank the candidates.
        if strategy == 'dense': ranked = [(id_, 1/(60+i)) for i,(id_,_) in enumerate(dense,1)]
        elif strategy == 'bm25': ranked = [(id_, 1/(60+i)) for i,(id_,sc) in enumerate(sparse,1) if sc > 0]
        else: ranked = rrf([(id_, 1.0) for id_,_ in dense], [(id_,sc) for id_,sc in sparse if sc > 0])
        ranked = [(id_,score) for id_,score in ranked if id_ in by_id]
        # 启用重排时：用 cross-encoder 对 query-片段对打分并重新排序
        if self.settings.rerank_enabled and ranked:
            if self._reranker is None:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder(self.settings.rerank_model,
                                              cache_folder=str(self.settings.embedding_cache))
            pairs = [(query,by_id[id_]['text']) for id_,_ in ranked]
            logits = self._reranker.predict(pairs)
            ranked = sorted(zip([id_ for id_,_ in ranked],[float(x) for x in logits]),
                            key=lambda x:x[1],reverse=True)
        # 组装最终输出，附带来源定位、排名与得分
        output = []
        for rank,(id_,score) in enumerate(ranked[:top_k],1):
            item = by_id[id_]
            output.append({'id':id_, 'doc_id':item['doc_id'], 'filename':item['metadata']['filename'],
                           'chunk_index':item['idx'], 'text':item['text'],
                           'start_char':item['metadata']['start_char'],
                           'end_char':item['metadata']['end_char'],
                           'rank':rank,'score':round(float(score),6)})
        return output