"""复用之前的 Document 与切分器（*在与原仓库合并时*）。
这里的回退实现允许该增量补丁独立运行，但不能替代原始的 pembelajaran 文件。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

try:
    # 优先使用原仓库根目录下的 Document 与 split_document
    from document import Document  # 原仓库根目录
    from text_splitter import split_document
    ORIGINAL_SPLITTER_USED = True
except ImportError:
    # 独立运行时启用内置回退实现
    ORIGINAL_SPLITTER_USED = False

    @dataclass
    class Document:
        """简化的文档对象：正文内容加元数据。"""
        page_content: str
        metadata: dict[str, Any] = field(default_factory=dict)

    def split_document(document: Document, chunk_size: int, chunk_overlap: int) -> list[Document]:
        """按固定窗口滑动切分文本，相邻块保留指定重叠。"""
        if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
            raise ValueError('Invalid chunk settings')
        chunks = []
        text = document.page_content
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            # 记录块序号与字符起止位置，便于后续溯源
            meta = {**document.metadata, 'chunk_index': len(chunks),
                    'start_char': start, 'end_char': end}
            chunks.append(Document(page_content=text[start:end], metadata=meta))
            if end == len(text):
                break
            start = end - chunk_overlap  # 下一块起点回退一个重叠长度
        return chunks