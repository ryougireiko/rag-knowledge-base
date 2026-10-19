"""可互换的 SQLite 演示型 / Chroma 持久化向量索引。
稀疏得分与语义得分刻意保持区分；融合时只使用排名。
"""
from __future__ import annotations
import sqlite3
import json
import math
from threading import RLock
from pathlib import Path


def cosine(a: list[float], b: list[float]) -> float:
    """计算两个向量的余弦相似度；维度不符或存在零向量时返回 0。"""
    if len(a) != len(b):
        return 0.0
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    return sum(x*y for x,y in zip(a,b))/(na*nb) if na and nb else 0.0


class SQLiteVectorIndex:
    """小规模离线/测试后端：线性扫描；并非大规模向量数据库。"""
    def __init__(self, path: Path):
        self.lock = RLock()
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        with self.lock, self.conn:
            self.conn.execute('''CREATE TABLE IF NOT EXISTS vectors (
                id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, vec TEXT NOT NULL)''')

    def upsert(self, chunks: list[dict], vectors: list[list[float]]) -> None:
        """批量插入或替换向量，向量以 JSON 字符串存储。"""
        assert len(chunks) == len(vectors)
        with self.lock, self.conn:
            self.conn.executemany('INSERT OR REPLACE INTO vectors VALUES (?,?,?)',
                [(c['id'], c['doc_id'], json.dumps(vec)) for c, vec in zip(chunks, vectors)])

    def search(self, vector: list[float], limit: int) -> list[tuple[str, float]]:
        """全表扫描计算余弦相似度，返回相似度最高的 limit 个 id。"""
        with self.lock:
            rows = self.conn.execute('SELECT id,vec FROM vectors').fetchall()
        ranked = [(row[0], cosine(vector, json.loads(row[1]))) for row in rows]
        return sorted(ranked, key=lambda x: x[1], reverse=True)[:limit]

    def delete(self, doc_id: str) -> None:
        """删除指定文档的全部向量。"""
        with self.lock, self.conn:
            self.conn.execute('DELETE FROM vectors WHERE doc_id=?', (doc_id,))

    def close(self):
        """关闭数据库连接。"""
        with self.lock:
            self.conn.close()


class ChromaVectorIndex:
    """基于 Chroma 的持久化向量索引，使用余弦距离的 HNSW 索引。"""
    def __init__(self, path: Path):
        import chromadb
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_or_create_collection(
            name='enterprise_knowledge_v2', configuration={'hnsw': {'space': 'cosine'}})

    def upsert(self, chunks: list[dict], vectors: list[list[float]]) -> None:
        """批量写入向量、原文与元数据。"""
        if not chunks: return
        self.collection.upsert(ids=[c['id'] for c in chunks], embeddings=vectors,
                               documents=[c['text'] for c in chunks],
                               metadatas=[{'doc_id': c['doc_id'], 'chunk_index': c['idx']}
                                          for c in chunks])

    def search(self, vector: list[float], limit: int) -> list[tuple[str,float]]:
        """查询最相近向量，将余弦距离转换为相似度 1-distance。"""
        total = self.collection.count()
        if total == 0: return []
        response = self.collection.query(query_embeddings=[vector], n_results=min(limit, total),
                                         include=['distances'])
        return [(i, 1.0-float(dist)) for i,dist in zip(response['ids'][0], response['distances'][0])]

    def delete(self, doc_id: str) -> None:
        """按元数据 doc_id 删除对应向量。"""
        self.collection.delete(where={'doc_id': doc_id})

    def close(self):
        # Chroma 客户端无需显式关闭
        pass


def make_vector(settings):
    """根据配置创建向量后端：chroma 或默认的 SQLite。"""
    if settings.vector_backend == 'chroma':
        return ChromaVectorIndex(settings.chroma_path)
    return SQLiteVectorIndex(settings.data_dir / 'vectors.sqlite3')