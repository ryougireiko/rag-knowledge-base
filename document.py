from dataclasses import dataclass, field
from typing import Any


@dataclass # 等于一个装饰器，帮忙自动实现下方我所要填入表格的功能的功能 不改变原代码的同时偷偷加了功能只作用于紧挨着它下面的那一个类。
class Document:
    """表示一个文档或文档片段的数据结构。

    Attributes:
        page_content: 文档的实际文本内容
        metadata: 包含文档来源、位置等信息的字典
    """
    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict) #类型字典，键是字符串，值可以是其他 后面这是生成了一个新字典
    # 如果直接用{}空字典去写，所有实例可能用同一个字典去了，所以要用field。