"""将虚构的公司文档批量导入到*当前*配置的索引中。"""
from pathlib import Path
from enterprise.service import Services


def main():
    """遍历 sample_company 下的 Markdown 文件逐个上传入库。"""
    svc=Services()
    try:
        # 定位 sample_company 目录并按文件名排序导入
        for p in sorted((Path(__file__).resolve().parents[1]/'sample_company').glob('*.md')):
            print(svc.ingest.upload(p.name,p.read_bytes()))
        print('Imported',len(svc.store.docs()),'example documents')
    finally: svc.close()

if __name__ == '__main__': main()