"""生产级 BGE 语义嵌入，以及一个明确标注为非语义、用于测试的确定性哈希嵌入。"""
from __future__ import annotations
import hashlib
import math
import re
from threading import Lock


def text_tokens(text: str) -> list[str]:
    """文本分词：提取英文/数字词，中文按双字滑切切分，用于特征哈希。"""
    english = re.findall(r'[a-zA-Z0-9]+', text.lower())
    chinese = re.findall(r'[\u4e00-\u9fff]+', text)
    return english + [part[i:i+2] for part in chinese for i in range(max(0,len(part)-1))]


class HashEmbeddings:
    """仅供演示：确定性特征哈希；不可与语义 BGE 相提并论。"""
    def __init__(self, dims: int = 256):
        self.dims = dims

    def encode(self, texts: list[str], query: bool = False) -> list[list[float]]:
        """将文本编码为固定维度的哈希特征向量并做 L2 归一化。"""
        results = []
        for text in texts:
            vec = [0.0] * self.dims
            for token in text_tokens(text):
                digest = hashlib.sha256(token.encode()).digest()
                slot = int.from_bytes(digest[:4], 'big') % self.dims  # 映射到桶位置
                sign = 1 if digest[4] % 2 else -1  # 决定 +/- 符号
                vec[slot] += sign
            norm = math.sqrt(sum(x*x for x in vec)) or 1.0  # 归一化除数，避免除零
            results.append([x/norm for x in vec])
        return results


class BGEEmbeddings:
    """生产级 BGE 语义嵌入，基于 sentence-transformers，模型懒加载。"""
    def __init__(self, model: str, cache_dir: str):
        self.name = model
        self.cache_dir = cache_dir
        self._model = None  # 模型延迟加载，首次编码时初始化
        self.lock = Lock()  # 保证模型加载的线程安全

    def encode(self, texts: list[str], query: bool = False) -> list[list[float]]:
        """编码文本为语义向量；检索查询时添加中文指令前缀。"""
        if not texts:
            return []
        with self.lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer
                from pathlib import Path

                # 优先使用本地已缓存的 BGE 模型，避免无网络环境下重复请求 Hugging Face
                cache_root = Path(self.cache_dir)
                model_cache = cache_root / f"models--{self.name.replace('/', '--')}" / "snapshots"

                if model_cache.is_dir():
                    snapshots = sorted(
                        (p for p in model_cache.iterdir() if p.is_dir()),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                else:
                    snapshots = []

                if snapshots:
                    self._model = SentenceTransformer(
                        str(snapshots[0]),
                        device="cpu",
                        local_files_only=True,
                    )
                else:
                    self._model = SentenceTransformer(
                        self.name,
                        cache_folder=self.cache_dir,
                    )
            # query=True 时加上 BGE 检索指令前缀，提升检索效果
            contents = [('为这个句子生成表示以用于检索相关文章：' + s) if query else s for s in texts]
            vectors = self._model.encode(contents, normalize_embeddings=True, batch_size=16)
            return vectors.tolist()


def make_embedder(settings):
    """根据配置创建嵌入后端：hash 走演示哈希，否则使用 BGE。"""
    if settings.embedding_backend == 'hash':
        return HashEmbeddings()
    return BGEEmbeddings(settings.embedding_model, str(settings.embedding_cache))