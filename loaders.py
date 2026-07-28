"""文档加载模块。

支持从本地文件系统加载 TXT 和 Markdown 文档，
并将其转换为统一的 Document 数据结构。
"""

from pathlib import Path
from typing import List
import re # 正则表达式模块 用来描述字符串匹配模式的语言。
from document import Document


def clean_text(text: str) -> str:
    """清洗文本内容。

    执行基础清洗操作：
    - 统一换行符为 \n
    - 去除首尾空白
    - 压缩连续空行（3个及以上空行压缩为2个换行）
    """
    # 统一换行符（Windows \r\n → \n）
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # 不能反过来用，不然就会把\r\n变成\n\n了

    # 去除首尾空白
    text = text.strip()

    # 压缩连续空行：3个及以上空行 → 2个换行

    text = re.sub(r'\n{3,}', '\n\n', text) # re正则表达式，定义了这种{3,}是3个以上的规则

    return text


def load_text_file(file_path: str | Path) -> Document:
    """加载单个文本文件并返回 Document 对象。传path对象或者字符串都行
     这就是我们要的那个document对象
    支持的格式：.txt, .md, .markdown
    """
    file_path = Path(file_path) # 转换为Path对象

    # 检查文件扩展名
    supported_extensions = {'.txt', '.md', '.markdown'}
    if file_path.suffix.lower() not in supported_extensions:
        raise ValueError(f"不支持的文件格式: {file_path.suffix}")

    # 读取文件
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"文件不存在: {file_path}")
    except Exception as e:
        raise IOError(f"读取文件失败 {file_path}: {e}")

    # 空文件处理：直接跳过  为了不让程序中断，反正这个空白的也对截取没有什么影响
    if not content.strip():
        print(f"警告: 文件 {file_path.name} 为空，将跳过")
        return Document(
            page_content="",
            metadata={ # 实例化了里面的字典
                "source": str(file_path),
                "filename": file_path.name,
                "file_type": file_path.suffix.lower()
            }
        )

    # 清洗文本
    cleaned_content = clean_text(content) # 清洗函数

    # 创建 Document
    return Document(
        page_content=cleaned_content,
        metadata={
            "source": str(file_path),
            "filename": file_path.name,
            "file_type": file_path.suffix.lower()
        }
    )


def load_documents(data_dir: str | Path) -> List[Document]:
    """从指定目录加载所有支持的文档。

    遍历目录，加载支持的格式，跳过不支持的格式和空文件。
    单个文件失败不影响其他文件。
    """
    data_dir = Path(data_dir) # 将里面的路径转换为Path对象

    # 检查目录是否存在
    if not data_dir.exists(): # 检查目录是否存在
        raise FileNotFoundError(f"目录不存在: {data_dir}")

    if not data_dir.is_dir():
        raise NotADirectoryError(f"路径不是目录: {data_dir}")

    documents = []
    supported_extensions = {'.txt', '.md', '.markdown'}

    # 遍历目录
    for file_path in data_dir.iterdir(): # 扫描这个data_dir这个文件夹底下的所有文件
        if not file_path.is_file(): # 如果不是文件，跳过
            continue

        # 跳过不支持的格式
        if file_path.suffix.lower() not in supported_extensions:
            print(f"跳过不支持的文件: {file_path.name}")
            continue

        try:
            doc = load_text_file(file_path) # 用上面那个函数把这个东西变成文字部分
            # 跳过空文档 如果是空的话就是false了，底下的这个就不会被执行
            if doc.page_content:
                documents.append(doc)
        except Exception as e:
            print(f"加载文件失败 {file_path.name}: {e}")
            continue

    return documents