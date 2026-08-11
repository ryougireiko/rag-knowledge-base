"""RAG 知识库处理主程序。

从 data 目录加载文档，切分成 Chunk，
生成 Embedding 向量，支持交互式语义检索。
"""

import numpy as np

from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from config import EMBEDDING_MODEL_NAME, QUERY_INSTRUCTION, TOP_K
from loaders import load_documents
from text_splitter import split_documents
from embeddings import EmbeddingModel
from retriever import retrieve_top_k


def main():
    """主流程：加载 → 切分 → 嵌入 → 检索"""

    print("=" * 60)
    print("RAG 知识库 - 语义检索")
    print("=" * 60)

    # ============================================
    # 启动时执行一次
    # ============================================

    # 1. 加载文档
    print("\n加载文档...")
    try:
        documents = load_documents(DATA_DIR)
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"错误: {e}")
        return

    print(f"成功加载 {len(documents)} 个文档")

    if not documents:
        print("没有找到可处理的文档，程序结束")
        return

    for doc in documents:
        print(f"   - {doc.metadata['filename']} ({len(doc.page_content)} 字符)")

    # 2. 切分文档
    print(f"\n切分文档 (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    chunks = split_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"生成 {len(chunks)} 个 Chunks")

    # 3. 初始化模型并生成 Chunk Embeddings
    print(f"\n加载 Embedding 模型...")
    print(f"  模型: {EMBEDDING_MODEL_NAME}")

    embedding_model = EmbeddingModel(
        model_name=EMBEDDING_MODEL_NAME,
        query_instruction=QUERY_INSTRUCTION
    )

    print("生成 Chunk 向量...")
    chunk_vectors = embedding_model.embed_documents(chunks)
    print(f"  Chunk 向量矩阵 shape: {chunk_vectors.shape}")
    print(f"  每个向量维度: {chunk_vectors.shape[1]}")

    # ============================================
    # 交互式检索循环
    # ============================================
    print("\n" + "=" * 60)
    print(f"开始交互式检索 (Top-K={TOP_K})")
    print("输入问题后返回最相关的 Chunk")
    print("输入 'exit' 退出")
    print("=" * 60)

    while True:
        # 读取用户问题
        query = input("\n🔍 问题: ").strip()

        # 退出
        if query.lower() == "exit":
            print("再见！")
            break

        # 空输入跳过
        if not query:
            print("问题不能为空，请重新输入")
            continue

        # 生成查询向量
        try:
            query_vector = embedding_model.embed_query(query)
        except ValueError as e:
            print(f"错误: {e}")
            continue

        # 检索 Top-K
        results = retrieve_top_k(
            query_vector=query_vector,
            documents=chunks,
            document_vectors=chunk_vectors,
            top_k=TOP_K
        )

        # 显示结果
        print(f"\n{'─' * 60}")
        print(f"  Top-{TOP_K} 检索结果")
        print(f"{'─' * 60}")

        for result in results:
            print(f"\n  Rank: {result.rank}")
            print(f"  Score: {result.score:.4f}")
            print(f"  来源: {result.document.metadata['filename']}")
            print(f"  Chunk: #{result.document.metadata['chunk_index'] + 1}")
            print(f"  位置: {result.document.metadata['start_char']} - "
                  f"{result.document.metadata['end_char']}")

            # 内容预览
            preview = result.document.page_content[:200]
            if len(result.document.page_content) > 200:
                preview += "..."
            print(f"  内容预览: {preview}")


if __name__ == "__main__":
    main()