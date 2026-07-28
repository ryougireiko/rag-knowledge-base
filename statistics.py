"""统计模块。"""

from typing import Dict, List # 类型注解模块
from document import Document


def get_length_stats(documents: List[Document]) -> Dict:
    if not documents: # 没有chunk的时候直接返回全0，防止后面min报错
        return {
            "total_chunks": 0,
            "min_length": 0,
            "max_length": 0,
            "avg_length": 0.0,
            "by_source": {}
        }

    lengths = [len(doc.page_content) for doc in documents] # 数了documents中每个doc中的content的长度
    # 这个就是把里面的每个chunk赋值给叫doc的东西，然后doc调用里面的page
    by_source = {} # 准备空字典
    for doc in documents:
        source = doc.metadata.get("filename", "unknown") # 如果键值不存在就返回unknown，查找文件的名字
        by_source[source] = by_source.get(source, 0) + 1 # 这里文件的名字塞进去，然后默认是0，第一次搞到所以加1，每次搞到就加1
        # 而且第一次塞进去就是建立了键值对
    return {
        "total_chunks": len(documents),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "avg_length": round(sum(lengths) / len(lengths), 1),
        "by_source": by_source
    }