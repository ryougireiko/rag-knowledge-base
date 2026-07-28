# RAG Knowledge Base

一个从底层逐步实现的本地 RAG（Retrieval-Augmented Generation）知识库项目。

当前版本完成了 RAG 数据处理链路的第一阶段：读取本地 TXT / Markdown 文档，完成基础文本清洗，将文档切分为带重叠区域的 Chunk，并为每个 Chunk 保存来源、编号和字符范围等 Metadata。

项目暂未接入大语言模型、Embedding 或向量数据库，重点是先理解并实现 RAG 系统中最基础的数据入口。

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
长度与来源统计
```

后续将继续扩展：

```text
Chunk
   ↓
Embedding
   ↓
向量相似度计算
   ↓
Top-K 检索
   ↓
LLM 生成回答
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
- 输出 Chunk 数量、长度和来源统计
- 预览前几个 Chunk 的来源、范围和内容
- 使用 pytest 覆盖文档加载、文本清洗和切片逻辑

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
│   └── test_text_splitter.py  # Chunk 切片测试
├── config.py                  # 数据目录与切片参数
├── document.py                # Document 数据结构
├── loaders.py                 # 文档读取与文本清洗
├── main.py                    # 项目运行入口
├── statistics.py              # Chunk 统计
├── text_splitter.py           # 字符级文本切片
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

## 配置

在 `config.py` 中修改：

```python
DATA_DIR = "data"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
```

参数说明：

| 参数 | 说明 |
| --- | --- |
| `DATA_DIR` | 本地知识文档所在目录 |
| `CHUNK_SIZE` | 单个 Chunk 的最大字符数 |
| `CHUNK_OVERLAP` | 相邻 Chunk 的重叠字符数 |

`CHUNK_OVERLAP` 必须小于 `CHUNK_SIZE`。

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/ryougireiko/rag-knowledge-base.git
cd rag-knowledge-base
```

### 2. 准备文档

将 UTF-8 编码的文本文件放入 `data/` 目录。

当前支持：

```text
.txt
.md
.markdown
```

### 3. 运行程序

```bash
python main.py
```

示例输出：

```text
============================================================
RAG 知识库 - 文档处理
============================================================
加载文档...
成功加载 2 个文档
切分文档 (chunk_size=500, overlap=50)...
生成 5 个 Chunks

Chunk 统计：
   总数量: 5
   最短: 218 字符
   最长: 500 字符
   平均: 419.0 字符

来源统计：
   sample.md: 3 个 Chunk
   sample.txt: 2 个 Chunk
```

程序还会打印前 3 个 Chunk 的来源文件、字符范围、长度和部分正文预览。

## 运行测试

安装 pytest：

```bash
pip install pytest
```

运行全部测试：

```bash
python -m pytest -q
```

当前测试覆盖：

- 换行符统一
- 首尾空白清理
- 连续空行压缩
- TXT 文档加载
- Markdown 文档加载
- Metadata 字段
- 文件和目录异常处理
- 不支持格式跳过
- 空文件跳过
- 短文本切片
- 长文本多 Chunk 切片
- Chunk overlap
- 最后一段文本完整性
- Chunk 编号连续性
- Metadata 保留
- 字符范围正确性
- 非法参数校验
- 空文本处理
- 死循环防护
- 多文档批量切片

当前共有 23 个测试用例。

## 当前限制

- 尚未调用大语言模型
- 尚未接入 Embedding 模型
- 尚未使用 FAISS、Chroma 或其他向量数据库
- 尚未实现语义相似度检索
- 尚未实现 Top-K Retriever
- 尚未实现最终知识库问答
- 当前只支持 TXT 和 Markdown
- 当前采用基础字符级切片，尚未按段落、句子或 Token 切片
- 当前仅遍历数据目录的第一层文件

## 后续计划

- [ ] 接入 Embedding 模型
- [ ] 实现余弦相似度计算
- [ ] 实现 Top-K Chunk 检索
- [ ] 输出检索分数和来源信息
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
5. Embedding 和向量检索真正接收的输入是什么

在理解底层流程后，再逐步接入 LangChain、向量数据库、FastAPI 和部署方案。
