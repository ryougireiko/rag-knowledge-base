"""文本切片模块。

将 Document 按照指定大小和重叠度切分成多个 Chunk，
每个 Chunk 保留原始文档的元数据信息并添加位置信息。
"""

from typing import List

from document import Document


def split_document(
        document: Document,
        chunk_size: int,
        chunk_overlap: int
) -> List[Document]:
    """将单个文档切分成多个 Chunk。

    使用字符级切片，按照 chunk_size 切分文本，
    Chunk 之间保持 chunk_overlap 的重叠。
    """
    # 参数验证
    if chunk_size <= 0:
        raise ValueError(f"chunk_size 必须大于 0，当前值: {chunk_size}")

    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap 不能为负数，当前值: {chunk_overlap}")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) 必须小于 chunk_size ({chunk_size})"
        )

    text = document.page_content # 取出正文，不用copy是因为字符串是不可变对象，而字典是可变对象
    original_metadata = document.metadata.copy() #复制元数据，防止.copy更改原数据

    # 处理空文本
    if not text:
        return [] # 返回一个空列表，结束函数

    chunks = []
    start = 0
    text_length = len(text) # 计算文本长度

    while start < text_length: # 切片循环
        # 确定当前 Chunk 的结束位置
        end = min(start + chunk_size, text_length) # 取最小值，防止越界

        # 提取 Chunk 文本
        chunk_text = text[start:end]

        # 创建 Chunk 的元数据（原始元数据 + 位置信息）
        chunk_metadata = original_metadata.copy() # 复制元数据，防止篡改了
        chunk_metadata.update({ # 每次更新这个metadata的信息，这个是添加了
            "chunk_index": len(chunks), # update是给字典里加键值对
            "start_char": start,
            "end_char": end
        })

        # 创建 Chunk Document
        chunk_doc = Document( # 这是一个document对象，每一个chunk是一个document对象
            page_content=chunk_text,
            metadata=chunk_metadata
        )
        chunks.append(chunk_doc) # append给列表后面加一个信息

        # 如果已经到达文本末尾，结束循环
        if end >= text_length:
            break

        # 计算下一个 Chunk 的起始位置
        start = end - chunk_overlap

        # 防止死循环：确保起始位置有前进
        if start <= chunks[-1].metadata["start_char"]:
            start = chunks[-1].metadata["start_char"] + 1

    return chunks


def split_documents(
        documents: List[Document],
        chunk_size: int,
        chunk_overlap: int
) -> List[Document]:
    """将多个文档切分成 Chunk。

    遍历文档列表，对每个文档调用 split_document 进行切分。
    """
    all_chunks = []

    for doc in documents:
        chunks = split_document(doc, chunk_size, chunk_overlap)
        all_chunks.extend(chunks) # append是把整个当一块塞进去，这个是把原来的列表拆了在一起

    return all_chunks