"""RAG 知识库处理主程序。

从 data 目录加载文档，切分成 Chunk，
输出统计信息和示例 Chunk 预览。
"""

from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from loaders import load_documents
from text_splitter import split_documents
from statistics import get_length_stats


def main():
    """主流程：加载 → 切分 → 统计 → 预览"""

    print("=" * 60)
    print("RAG 知识库 - 文档处理")
    print("=" * 60)

    # 1. 加载文档
    print("加载文档...")
    try:
        documents = load_documents(DATA_DIR) # 返回了一个叫做documents的列表，里面分别是document那个类的实例化
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"错误: {e}")
        return

    print(f"成功加载 {len(documents)} 个文档")

    if not documents:
        print("没有找到可处理的文档，程序结束")
        return

    for doc in documents:
        print(f"   - {doc.metadata['filename']} ({len(doc.page_content)} 字符)") # 打印出文档的文件名和字符长度

    # 2. 切分文档
    print(f"切分文档 (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    chunks = split_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP) # 切分多个文档的函数
    print(f"生成 {len(chunks)} 个 Chunks")

    # 3. 统计结果
    print("Chunk 统计：")
    stats = get_length_stats(chunks) # 统计切分的结果
    print(f"   总数量: {stats['total_chunks']}")
    print(f"   最短: {stats['min_length']} 字符")
    print(f"   最长: {stats['max_length']} 字符")
    print(f"   平均: {stats['avg_length']} 字符")

    print("来源统计：")
    for source, count in stats['by_source'].items():
        print(f"   {source}: {count} 个 Chunk")

    # 4. 打印前 3 个 Chunk 预览
    preview_count = min(3, len(chunks))
    print(f"前 {preview_count} 个 Chunk 预览：")
    print("-" * 60)

    for i, chunk in enumerate(chunks[:preview_count]): # i就是编号，这个是
        print(f"\nChunk #{chunk.metadata['chunk_index'] + 1}") # 取chunk的序号
        print(f"  来源: {chunk.metadata['filename']}") # 打印名字
        print(f"  位置: {chunk.metadata['start_char']} - {chunk.metadata['end_char']}") # 位置
        print(f"  长度: {len(chunk.page_content)} 字符") # 内容长度

        preview_text = chunk.page_content[:150] # 预览一部分
        if len(chunk.page_content) > 150:
            preview_text += "..."
        print(f"  内容预览: {preview_text}")
        print("-" * 60)

    print(f"处理完成！共处理 {len(documents)} 个文档，生成 {len(chunks)} 个 Chunk")


if __name__ == "__main__":
    main()