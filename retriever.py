import numpy as np
from dataclasses import dataclass


@dataclass #它的作用是自动为类生成常用方法，让你用更简洁的代码定义数据容器。
class SearchResult: # 一个类，自动生成了很多方法
    """检索结果"""
    document: object   # Document 对象（Chunk）
    score: float       # 余弦相似度分数
    rank: int          # 排名（从 1 开始）
# 这样可以让这三个返回在一起，如果用元组的话太乱了

def cosine_similarity(query_vector, document_vectors):
    """
    手写余弦相似度计算

    Args:
        query_vector: 查询向量，一维 shape (dim,)
        document_vectors: 文档向量矩阵，二维 shape (n, dim)

    Returns:
        numpy.ndarray: 每个文档的相似度分数 shape (n,)
    """
    # 1. 检查查询向量是一维
    if query_vector.ndim != 1: # 查询向量必须是 (512,)，不能是 (1, 512)
        raise ValueError(f"query_vector 必须是一维，当前维度: {query_vector.ndim}")

    # 2. 检查文档向量是二维
    if document_vectors.ndim != 2: # 文档向量必须是 (n, 512)，不能是 (512,) 这个2是2维数组
        raise ValueError(f"document_vectors 必须是二维，当前维度: {document_vectors.ndim}")

    # 3. 检查维度一致
    if query_vector.shape[0] != document_vectors.shape[1]: # 矩阵乘法
        raise ValueError(
            f"向量维度不一致: query={query_vector.shape[0]}, "
            f"documents={document_vectors.shape[1]}"
        )

    # 4. 查询零向量报错
    if np.all(query_vector == 0):
        raise ValueError("查询向量不能为零向量")
    # 点积 A · B = 0（任何向量与零向量的点积都是0）
    # 模长 ||A|| = 0（零向量的长度是0）
    # 分母 0 × ||B|| = 0
    # 结果 0 / 0 → 数学上无意义（NaN 或报错）

    # 5. 计算余弦相似度
    # 公式: cos = (A·B) / (||A|| * ||B||)
    query_norm = np.linalg.norm(query_vector) # 查询向量的模长
    doc_norms = np.linalg.norm(document_vectors, axis=1) # # 每个文档向量的模长（n个值）

    # 点积
    dot_products = np.dot(document_vectors, query_vector)

    # 避免文档零向量导致除零
    doc_norms = np.where(doc_norms == 0, 1, doc_norms)

    similarities = dot_products / (query_norm * doc_norms)

    return similarities


def retrieve_top_k(query_vector, documents, document_vectors, top_k):
    """
    Top-K 检索

    Args:
        query_vector: 查询向量
        documents: Document 对象列表
        document_vectors: 文档向量矩阵
        top_k: 返回前 K 个

    Returns:
        list[SearchResult]
    """
    # 参数校验
    if top_k <= 0:
        raise ValueError(f"top_k 必须大于 0，当前值: {top_k}")

    if len(documents) != document_vectors.shape[0]:
        raise ValueError(
            f"文档数量({len(documents)})与向量数量"
            f"({document_vectors.shape[0]})不一致"
        )

    if not documents:
        return []

    # 1. 计算所有相似度
    scores = cosine_similarity(query_vector, document_vectors)

    # 2. 按分数降序排列（加负号让升序变降序）
    sorted_indices = np.argsort(-scores, kind="stable")

    # 3. 取前 K 个
    top_k = min(top_k, len(documents))
    top_indices = sorted_indices[:top_k]

    # 4. 构造结果
    results = []
    for rank, idx in enumerate(top_indices, start=1):
        result = SearchResult(
            document=documents[idx],
            score=float(scores[idx]),
            rank=rank
        )
        results.append(result)

    return results