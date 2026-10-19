"""仅从环境变量加载配置，不将任何密钥提交到 Git。"""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # 加载 .env 文件中的环境变量


def yes(key: str, default: str = 'false') -> bool:
    """读取布尔型环境变量，识别 1/yes/true/on 为真。"""
    return os.getenv(key, default).strip().lower() in {'1', 'yes', 'true', 'on'}


@dataclass(frozen=True)
class Settings:
    """不可变的应用配置项集合。"""
    data_dir: Path          # 数据根目录
    sqlite_path: Path       # 文档元数据库路径
    checkpoint_path: Path   # Agent 检查点数据库路径
    chroma_path: Path       # Chroma 向量库路径
    embedding_cache: Path   # 嵌入模型缓存目录
    embedding_model: str    # 嵌入模型名称
    embedding_backend: str  # 嵌入后端（hash/bge）
    vector_backend: str     # 向量库后端（sqlite/chroma）
    llm_backend: str        # 大模型后端（mock/deepseek）
    deepseek_api_key: str   # DeepSeek API 密钥
    deepseek_base_url: str  # DeepSeek 接口地址
    deepseek_model: str     # DeepSeek 模型名称
    api_token: str          # API 鉴权 Token
    max_upload_mb: int      # 上传文件大小上限（MB）
    chunk_size: int         # 文档切分块大小
    chunk_overlap: int      # 相邻块重叠长度
    top_k: int              # 检索返回片段数
    rerank_enabled: bool    # 是否启用重排
    rerank_model: str       # 重排模型名称
    max_agent_steps: int    # Agent 最大推理步数

    @classmethod
    def from_env(cls) -> 'Settings':
        """从环境变量构建配置，演示模式下使用轻量默认后端。"""
        root = Path(os.getenv('APP_DATA_DIR', './.local_data')).resolve()
        demo = yes('DEMO_MODE')  # 演示模式开关
        fields = cls(
            data_dir=root,
            sqlite_path=root / 'documents.sqlite3',
            checkpoint_path=root / 'agent_checkpoints.sqlite3',
            chroma_path=root / 'chroma',
            embedding_cache=Path(os.getenv('MODEL_CACHE_DIR', str(root / 'models'))).resolve(),
            embedding_model=os.getenv('EMBEDDING_MODEL', 'BAAI/bge-small-zh-v1.5'),
            embedding_backend=os.getenv('EMBEDDING_BACKEND', 'hash' if demo else 'bge').lower(),
            vector_backend=os.getenv('VECTOR_BACKEND', 'sqlite' if demo else 'chroma').lower(),
            llm_backend=os.getenv('LLM_BACKEND', 'mock' if demo else 'deepseek').lower(),
            deepseek_api_key=os.getenv('DEEPSEEK_API_KEY', ''),
            deepseek_base_url=os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com'),
            deepseek_model=os.getenv('DEEPSEEK_MODEL', 'deepseek-chat'),
            api_token=os.getenv('APP_API_TOKEN', ''),
            max_upload_mb=int(os.getenv('MAX_UPLOAD_MB', '8')),
            chunk_size=int(os.getenv('CHUNK_SIZE', '450')),
            chunk_overlap=int(os.getenv('CHUNK_OVERLAP', '60')),
            top_k=int(os.getenv('TOP_K', '5')),
            rerank_enabled=yes('RERANK_ENABLED'),
            rerank_model=os.getenv('RERANK_MODEL', 'BAAI/bge-reranker-base'),
            max_agent_steps=int(os.getenv('MAX_AGENT_STEPS', '12')),
        )
        # 校验切分参数合法性
        if fields.chunk_size <= 0 or not (0 <= fields.chunk_overlap < fields.chunk_size):
            raise ValueError('Invalid CHUNK_SIZE/CHUNK_OVERLAP')
        # 校验后端取值范围
        if fields.embedding_backend not in {'hash','bge'} or fields.vector_backend not in {'sqlite','chroma'}:
            raise ValueError('Invalid embedding/vector backend')
        if fields.llm_backend not in {'mock','deepseek'}:
            raise ValueError('Invalid LLM_BACKEND')
        # 校验上传与检索限制
        if fields.max_upload_mb <= 0 or not (1 <= fields.top_k <= 20):
            raise ValueError('Invalid upload/query limits')
        fields.data_dir.mkdir(parents=True, exist_ok=True)  # 确保数据目录存在
        return fields