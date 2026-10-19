"""RAG 服务：证据优先构建上下文，并提供可溯源的来源编号。"""
from __future__ import annotations


class RAGService:
    """RAG 服务：组合检索器与大模型，输出带来源的回答。"""
    def __init__(self, retriever, llm):
        self.retriever, self.llm = retriever, llm

    @staticmethod
    def context(hits: list[dict]) -> str:
        """将检索命中拼装为带 [S编号] 标记的上下文文本。"""
        return '\n\n'.join(f"[S{i}] filename={h['filename']} chunk={h['chunk_index']}\n"
                           f"{h['text'][:1100]}" for i,h in enumerate(hits,1))

    @staticmethod
    def sources(hits: list[dict]) -> list[dict]:
        """生成可溯源的来源列表：含文档、片段定位、得分与预览。"""
        return [{'source_id':f'S{i}', 'doc_id':h['doc_id'], 'filename':h['filename'],
                 'chunk_index':h['chunk_index'],'start_char':h['start_char'],
                 'end_char':h['end_char'], 'score':h['score'],
                 'preview':h['text'][:180]} for i,h in enumerate(hits,1)]

    def retrieve(self, query: str, top_k: int | None = None, strategy: str = 'hybrid'):
        """执行检索，返回命中结果与对应来源列表。"""
        hits = self.retriever.search(query, top_k=top_k, strategy=strategy)
        return hits, self.sources(hits)

    def answer(self, query: str, top_k: int | None = None, strategy: str = 'hybrid') -> dict:
        """完整问答：无命中时如实告知，否则基于上下文生成回答。"""
        hits, sources = self.retrieve(query, top_k, strategy)
        if not hits:  # 无证据时不臆造，直接返回提示
            return {'answer':'知识库没有找到相关资料，无法根据现有资料回答。', 'sources':[], 'strategy':strategy}
        return {'answer':self.llm.answer(query,self.context(hits)),
                'sources':sources,'strategy':strategy}