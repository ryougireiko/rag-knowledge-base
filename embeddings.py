from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingModel:
    """本地 Embedding 模型封装"""

    def __init__(self, model_name, query_instruction="", model=None):
        """
        Args:
            model_name: 模型名称，如 "BAAI/bge-small-zh-v1.5"
            query_instruction: 查询时添加的前缀
            model: 测试时可注入假模型，不传则加载真实模型，这个model是那个调用model的api
        """
        self.model_name = model_name
        self.query_instruction = query_instruction

        if model is not None: # 如果不是NONE的话就加载我们给的这个模型
            self.model = model
        else: # 如果没命名就默认用这个模型
            self.model = SentenceTransformer(model_name) # 将中文文本变成数字的工具

    def embed_documents(self, documents): # 文档向量化
        """批量生成文档 Chunk 的向量，不加查询前缀"""
        if not documents:
            return np.array([])

        texts = [doc.page_content for doc in documents] # 从切好的chunk里面把他们提取出来，最后做成一个列表

        embeddings = self.model.encode( # 调用了SentenceTransformer里的方法，最终吐出来一堆数字向量
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        return embeddings

    def embed_query(self, query): # 将用户的问题转成512维向量，加上查询前缀
        """生成查询向量，加查询前缀"""
        if not query or not query.strip(): # 输入了空文本
            raise ValueError("查询文本不能为空")

        if self.query_instruction: # 提示词加上用户的内容
            query = self.query_instruction + query

        embedding = self.model.encode( #这里这个按照规则，像这种一个的列表就是会返回一个一维的向量
            query,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        return embedding