"""离线确定性冒烟测试，不下载模型 / 不产生 DeepSeek 费用。
用法：先设置 DEMO_MODE=true，再运行 python -m scripts.smoke_demo
"""
import os
os.environ['DEMO_MODE']='true'  # 强制演示模式，走 mock 后端
from enterprise.service import Services
from pathlib import Path


def main():
    """上传差旅政策文档并进行一次 RAG 问答，验证检索链路可用。"""
    service=Services()
    try:
        # 定位示例差旅政策文件并入库
        path=Path(__file__).resolve().parent.parent / 'sample_company' / 'travel_policy.md'
        result=service.ingest.upload(path.name,path.read_bytes())
        print('Indexed:',result)
        # 提问并检查回答附带来源
        reply=service.rag.answer('上海普通员工住宿报销标准是什么？')
        print('Answer:',reply['answer'][:400]); print('Sources:',reply['sources'][:1])
        assert reply['sources']
        print('PASS: Offline smoke test (not an LLM/Agent acceptance test)')
    finally: service.close()

if __name__=='__main__': main()