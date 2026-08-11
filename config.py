DATA_DIR = "data" # 数据文档放在哪
CHUNK_SIZE = 500 # 每个分块的大小500字符
CHUNK_OVERLAP = 50 # 相邻两块的重叠
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5" # BAAI 模型发布方，bge BAAI GENERAL EMBEDDING，small轻量版，参数少，zh针对中文
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章：" # "为这个句子生成表示以用于检索相关文章：Embedding 是干什么用的？"
TOP_K = 3 # 返回相似度最高的3个文档 3 是一个常用经验值，在 "信息够用" 和 "不过载" 之间取平衡。