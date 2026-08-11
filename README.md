# RAG Knowledge Base

一个从底层逐步实现的本地 RAG（Retrieval-Augmented Generation）知识库项目。

当前版本已完成 RAG 数据处理链路的前两个阶段：文档加载与清洗、Chunk 切片，以及基于 BGE 模型的本地 Embedding 与内存语义检索。

项目暂未接入大语言模型或向量数据库，重点是先理解并手写实现 RAG 系统中最基础的数据入口和检索核心。

## 项目流程

```text
本地文档
   ↓
文档加载
   ↓
文本清洗
   ↓
字符级 Chunk 切片
   ↓
Metadata 管理
   ↓
Embedding 向量化（BGE Small 中文模型）
   ↓
手写余弦相似度计算
   ↓
Top-K 语义检索
```

后续将继续扩展：

```text
Embedding 向量存入向量数据库（FAISS / Chroma）
   ↓
用户提问 → 向量检索 → Top-K 结果
   ↓
LLM 生成回答
   ↓
FastAPI 查询接口
```

## 当前功能

- 支持加载 `.txt`、`.md` 和 `.markdown` 文件
- 使用 UTF-8 编码读取本地文档
- 统一不同操作系统的换行符
- 去除文本首尾空白
- 压缩三个及以上的连续换行
- 自动跳过不支持的文件格式
- 自动跳过空文件
- 单个文件加载失败时不影响其他文件
- 使用统一的 `Document` 数据结构保存正文和 Metadata
- 支持配置 `chunk_size` 和 `chunk_overlap`
- 实现字符级 Chunk 切片
- 校验非法切片参数，避免死循环
- 保留原始文档 Metadata，并增加 Chunk 位置信息
- ✅ 中文文本 Embedding（BAAI/bge-small-zh-v1.5）
- ✅ 512 维向量表示
- ✅ 查询向量与 Chunk 向量分离处理
- ✅ 手写余弦相似度（不依赖 sklearn）
- ✅ Top-K 语义检索
- ✅ 检索分数、排名与 Metadata 来源追踪
- ✅ CLI 交互式查询
- ✅ 完全离线测试（Fake 模型注入）
- 使用 pytest 覆盖文档加载、文本清洗、切片、Embedding 和检索逻辑

## 技术栈

- Python 3.10+
- dataclasses
- pathlib
- re
- pytest

项目核心运行逻辑只使用 Python 标准库，测试部分依赖 pytest。

## 项目结构

```text
rag-knowledge-base/
├── data/
│   ├── sample.md              # RAG 技术学习示例文档
│   └── sample.txt             # ChatBot PRO 项目示例文档
├── tests/
│   ├── test_loaders.py        # 文本清洗与文档加载测试
│   ├── test_text_splitter.py  # Chunk 切片测试
│   ├── test_embeddings.py     # Embedding 封装测试（Fake 模型）
│   └── test_retriever.py      # 余弦相似度与检索测试
├── config.py                  # 数据目录、切片参数与 Embedding 配置
├── document.py                # Document 数据结构
├── loaders.py                 # 文档读取与文本清洗
├── text_splitter.py           # 字符级文本切片
├── embeddings.py              # BGE Embedding 模型封装
├── retriever.py               # 余弦相似度与 Top-K 检索
├── statistics.py              # Chunk 统计（Day 11）
├── main.py                    # 语义检索交互入口
└── README.md
```

## 核心设计

### 1. Document 数据结构

项目使用 `dataclass` 定义统一的文档对象：

```python
@dataclass
class Document:
    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)
```

- `page_content`：保存原始文档或 Chunk 的正文
- `metadata`：保存来源文件、文件类型、Chunk 编号和字符范围

原始文档和切片后的 Chunk 使用同一种数据结构，便于后续接入 Embedding、向量存储和 Retriever。

### 2. 文档加载与清洗

`loaders.py` 提供：

```python
load_text_file(file_path)
load_documents(data_dir)
clean_text(text)
```

文档加载后会生成如下 Metadata：

```python
{
    "source": "data/sample.txt",
    "filename": "sample.txt",
    "file_type": ".txt"
}
```

### 3. Chunk 切片

当前使用基础字符级切片：

```text
chunk_size = 500
chunk_overlap = 50

Chunk 1：0 - 500
Chunk 2：450 - 950
Chunk 3：900 - 文本末尾
```

相邻 Chunk 保留一定的重叠内容，可以减少关键信息恰好处于切片边界时产生的上下文丢失。

每个 Chunk 会在原始 Metadata 基础上增加：

```python
{
    "chunk_index": 0,
    "start_char": 0,
    "end_char": 500
}
```

其中 `chunk_index` 是当前来源文档内部的 Chunk 编号。

### 4. 参数校验

切片函数会检查：

- `chunk_size` 必须大于 0
- `chunk_overlap` 不能小于 0
- `chunk_overlap` 必须小于 `chunk_size`
- 空文本返回空列表
- 最后一个 Chunk 必须覆盖到文本末尾
- 循环过程中起始位置必须持续向前移动

### 5. Embedding 模型封装

`embeddings.py` 封装了 BGE 中文 Embedding 模型：

python

```
class EmbeddingModel:
    def embed_documents(self, documents) -> np.ndarray:
        """批量生成 Chunk 向量（不加查询指令）"""
        ...

    def embed_query(self, query) -> np.ndarray:
        """生成查询向量（加查询指令前缀）"""
        ...
```



核心设计：

- SentenceTransformer 只初始化一次
- 查询指令（`QUERY_INSTRUCTION`）仅加在用户问题前，不加在文档 Chunk 上
- 使用 `normalize_embeddings=True` 输出归一化向量
- 测试时可通过 `model` 参数注入 Fake 模型，完全离线验证

### 6. 余弦相似度（手写）

`retriever.py` 中手写实现了余弦相似度公式：

python

```
cos = (A · B) / (||A|| × ||B||)
```



- 不依赖 sklearn 或 sentence-transformers 的封装
- 检查向量维度、零向量等边界条件
- 处理文档零向量不会导致除零崩溃

### 7. Top-K 语义检索

python

```
@dataclass
class SearchResult:
    document: Document  # 命中的 Chunk
    score: float        # 相似度分数
    rank: int           # 排名（从 1 开始）
```



检索流程：

text

```
查询向量 → 计算全部相似度 → 降序排序 → 取 Top-K → 构造 SearchResult
```



- 分数从高到低排列
- rank 从 1 开始
- 相同分数时使用稳定排序
- score 不写入原始 Document.metadata（属于检索结果而非 Chunk 属性）

## 配置

在 `config.py` 中修改：

```python
DATA_DIR = "data"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
TOP_K = 3
```

参数说明：

| 参数                   | 说明                    |
| --- | --- |
| `DATA_DIR`             | 本地知识文档所在目录    |
| `CHUNK_SIZE`           | 单个 Chunk 的最大字符数 |
| `CHUNK_OVERLAP`        | 相邻 Chunk 的重叠字符数 |
| `EMBEDDING_MODEL_NAME` | BGE Embedding 模型名称  |
| `QUERY_INSTRUCTION`    | 查询时添加的前缀指令    |
| `TOP_K`                | 返回的最相关 Chunk 数量 |

`CHUNK_OVERLAP` 必须小于 `CHUNK_SIZE`。

## 快速开始

### 1. 克隆项目

bash

```
git clone https://github.com/ryougireiko/rag-knowledge-base.git
cd rag-knowledge-base
```



### 2. 安装依赖

bash

```
pip install -r requirements.txt
```



### 3. 准备文档

将 UTF-8 编码的文本文件放入 `data/` 目录。

当前支持：

text

```
.txt
.md
.markdown
```



### 4. 运行语义检索

bash

```
python main.py
```



示例输出：

text

```
============================================================
RAG 知识库 - 语义检索
============================================================

加载文档...
成功加载 2 个文档
   - sample.md (1120 字符)
   - sample.txt (828 字符)

切分文档 (chunk_size=500, overlap=50)...
生成 5 个 Chunks

加载 Embedding 模型...
  模型: BAAI/bge-small-zh-v1.5
生成 Chunk 向量...
  Chunk 向量矩阵 shape: (5, 512)
  每个向量维度: 512

============================================================
开始交互式检索 (Top-K=3)
输入问题后返回最相关的 Chunk
输入 'exit' 退出
============================================================

🔍 问题: 什么是 Embedding

────────────────────────────────────────────────────────────
  Top-3 检索结果
────────────────────────────────────────────────────────────

  Rank: 1
  Score: 0.6448
  来源: sample.md
  Chunk: #3
  位置: 900 - 1120
  内容预览: 虽然目前还没有接入 Embedding 模型...
```



输入 `exit` 退出程序。

## 运行测试

bash

```
python -m pytest -q
```



当前测试覆盖：

- 换行符统一、首尾空白清理、连续空行压缩
- TXT / Markdown 文档加载、Metadata 字段
- 文件和目录异常处理、不支持格式跳过、空文件跳过
- 短文本切片、长文本多 Chunk 切片、Chunk overlap
- 最后一段文本完整性、Chunk 编号连续性
- Metadata 保留、字符范围正确性
- 非法参数校验、空文本处理、死循环防护
- 多文档批量切片
- ✅ Embedding 封装：维度检查、查询指令添加、空输入处理
- ✅ 余弦相似度：相同/垂直/相反向量、非归一化向量、维度校验
- ✅ Top-K 检索：降序排列、rank 编号、边界条件、Metadata 保留
- ✅ 完全离线（Embedding 测试使用 Fake 模型，不下载真实模型）

## 当前限制

- 尚未调用大语言模型
- 尚未生成最终回答
- 尚未使用 FAISS、Chroma 或其他向量数据库
- 向量仅保存在内存，程序重启需重新生成
- 尚未实现相似度阈值过滤
- 尚未实现 Reranker 重排序
- 尚未实现 FastAPI 查询接口
- 尚未实现 Docker 部署
- 当前只支持 TXT 和 Markdown
- 当前采用基础字符级切片，尚未按段落、句子或 Token 切片
- 当前仅遍历数据目录的第一层文件

## 后续计划

- [x] 接入 Embedding 模型
- [x] 实现余弦相似度计算
- [x] 实现 Top-K Chunk 检索
- [x] 输出检索分数和来源信息
- [ ] 接入向量数据库
- [ ] 组合检索结果与大语言模型
- [ ] 实现可溯源的 RAG 问答
- [ ] 使用 FastAPI 提供查询接口
- [ ] 增加 PDF 等文档格式
- [ ] 增加 Docker 部署

## 项目目标

本项目不是直接调用框架快速拼装一个知识库，而是先手写 RAG 的基础组件，理解以下过程：

1. 本地文档如何转换为统一的数据结构
2. 为什么长文档需要切分成 Chunk
3. Chunk overlap 如何减少边界信息丢失
4. Metadata 如何支持后续来源追踪
5. Embedding 如何将文本映射到语义向量空间
6. 为什么文档和查询必须使用同一个 Embedding 模型
7. 余弦相似度如何衡量向量间的语义距离
8. Top-K 检索的本质是"相对最相似"而非"绝对正确"
