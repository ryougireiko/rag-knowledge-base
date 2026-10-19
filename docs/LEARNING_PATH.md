# 项目反向学习手册：拿到代码后怎么学（工程视角）

## 优先目标

本项目代码是 AI 辅助生成的求职作品候选，必须亲自运行、测试和修改后才能把具体功能写在简历上。不要死背源码。每个阶段都回答：「输入 / 输出 / 为什么这样设计 / 出错怎么查 / 换方案的代价」。

### 任务 1：追踪一次文档上传（约 60～90 分钟）

从 `POST /api/documents` 开始，沿着 `api.py` -> `IngestionService.upload()` -> `parse_file()` -> `legacy_bridge.split_document()` -> `embedder.encode()` -> `vector.upsert()` -> SQLite `DocumentStore`。

自己修改：把 `MAX_UPLOAD_MB` 调成 1，上传大文件并观察 HTTP 413。再重复上传同一个文件，确认返回 `duplicate=true`。画出这一条链路。

### 任务 2：亲手讲明白检索（约 90 分钟）

重点读 `retrieval.py`：BM25、Dense Search、RRF、Reranker。问题：为何不同检索分数不能简单直接相加？RRF 为什么只依赖排名？过小的 chunk 对召回有什么影响？

自己修改：改成 `top_k=3`，分别调用 strategy `dense`/`bm25`/`hybrid`；记录来源和延迟，分析差异。额外设计一个含数字的测试问题。

### 任务 3：理解 RAG 生成与来源（约 60 分钟）

重点读 `rag.py` 和 `llm.py`。知道 Context 是检索结果而非整个知识库；知道 `[S1]` 的编号如何关联元数据；知道 Prompt 无法完全保证模型不幻觉。

自己修改：上传一段与你问题无关的文档，问一个知识库完全不知道的问题，检查回答是否承认不足。

### 任务 4：LangGraph Agent 工作流（约 2～3 小时）

重点读 `agent.py`：MessagesState、SystemMessage、ToolNode、tools_condition、SqliteSaver、thread_id、recursion_limit。

自己修改：让 Agent 同一个 `thread_id` 连续提问两次，验证会话状态。观察 `tool_trace`，说明哪一次调用由模型触发、工具如何返回，以及 MCP Client 调用 Server 的过程。

### 任务 5：FastAPI 与 Docker（约 90 分钟）

重点读 `api.py` 的 `lifespan`、Pydantic schema、`UploadFile`、SSE。看 `docker-compose.v2.yml` 是如何把 API 和 UI 连接的、哪里保存 Chroma/模型缓存、为什么 127.0.0.1 比对公网开放安全。

自己修改：更换错误的 API Token，观察 HTTP 401；构造非法 top_k，观察 HTTP 422；查看日志定位。

### 任务 6：跑真正的 Evaluation（约 90 分钟）

改用 BGE+Chroma，上传样本文件，执行 `python -m evaluation.run_eval`。分别记录 dense、bm25、hybrid。把数据写进 `docs/EVAL_RESULTS.md`；若 hybrid 没有提升，也要如实写明。

## 最重要的 10 道验收题

1. 上传新文档为何要生成稳定 Chunk ID？
2. 为什么 Embedding 的查询和文档应使用同一向量空间？
3. BM25 和向量检索擅长解决什么不同问题？
4. 为什么混合检索选 RRF 而非直接加分？
5. Reranker 何时值得使用，性能代价是什么？
6. 为什么文档来源必须在 Chunk Metadata 里保存？
7. 直接 RAG 与 Agent 调用搜索工具的差异是什么？
8. LangGraph 的 `thread_id` 和 SQLite Checkpoint 如何协作？
9. MCP 解决什么问题？本项目的 MCP Client 与 Server 怎么通信？
10. Demo/Mock 测试通过为什么不等于项目可以写「生产已上线」？

## 学习顺序建议

优先达到能运行和解释主链路，然后创建简历 V1 试投。之后根据岗位 JD 和面试追问回头补 Python 异步、HTTP、向量数据库、评测与系统设计。不要因为理论没学完而一直不投。
