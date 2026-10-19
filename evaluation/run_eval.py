"""仅评估检索效果：报告 Hits@K/MRR，*而非*虚构的质量提升。"""
from __future__ import annotations
import json
from pathlib import Path
from enterprise.service import Services


def evaluate(svc: Services, questions: list[dict], strategy: str, k: int=5) -> dict:
    """按指定策略评估：统计命中率(Hits@K)与平均倒数排名(MRR)。"""
    hits,rr = 0,0.0
    for row in questions:
        results=svc.retriever.search(row['query'],top_k=k,strategy=strategy)
        # 找到期望文件名首次出现的排名（从 1 开始），未命中则为 None
        rank=next((i for i,h in enumerate(results,1) if h['filename']==row['expected_filename']),None)
        hits+=bool(rank)
        rr+=(1/rank if rank else 0.0)
    return {'strategy':strategy,'n':len(questions),f'hit@{k}':hits/len(questions),
            f'mrr@{k}':rr/len(questions)}


if __name__=='__main__':
    # 加载评测问题集
    questions=json.loads((Path(__file__).parent/'questions.json').read_text(encoding='utf8'))
    svc=Services()
    try:
        # 样例文档不足时，提示先导入
        if len(svc.store.docs()) < 7:
            raise SystemExit('请先运行 python -m scripts.import_samples 导入全部样例文档')
        # 依次评估三种检索策略
        for method in ['dense','bm25','hybrid']:
            print(evaluate(svc,questions,method))
    finally: svc.close()