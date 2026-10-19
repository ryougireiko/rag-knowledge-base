# 2026 届 AI 大模型应用开发 — 简历项目素材（待本人实测后采用）

> ⚠️ 下列项目能力是**目标状态**，不是当前测试结果。没有亲自跑通的功能删掉，性能数字一律待实测填写。

## 基于 LangGraph + Advanced RAG 的企业知识库智能 Agent 系统

**技术栈**：Python、FastAPI、DeepSeek、LangChain/ LangGraph、Chroma、BGE、BM25、RRF、MCP、SQLite、Docker、pytest

**背景**：围绕模拟企业差旅制度和知识库治理场景，开发支持文档管理、证据检索、多工具任务处理和引用溯源的单用户智能助手。

**经历描述（目标版，实际测试成功才保留）**：

- 复用自研 Document 和文本切分模块，完成文档解析、Chunk 元数据管理、BGE 向量化和 Chroma 持久化，实现文本型 PDF/Markdown/TXT 的索引与去重。
- 实现 BM25 与向量检索混合召回，结合 RRF 融合与可选 CrossEncoder Reranker；构建固定评测集对比不同检索策略的 Hit@K、MRR 和延迟。
- 基于 LangGraph 构建有状态工具调用 Agent，集成知识库检索、日期查询与 MCP v2 业务计算工具，使用 SQLite Checkpoint 管理多轮会话状态。
- 使用 FastAPI 和 Streamlit 提供文件管理、RAG 问答及 Agent 任务交互，补充 pytest/Mock、SSE 流式输出、Docker Compose 和运行文档。

**可选量化结果（仅实测后填写）**：评测问题 N=[待填写]，Hybrid Hit@5=[待填写]，Dense Hit@5=[待填写]；单次查询 P50=[待填写]ms。*不必强求必须有提升*。

## ChatBot PRO（已有仓库）

描述重点：DeepSeek API、Streaming、对话历史、Pydantic 结构化输出、Tool Calling、Mock 测试、日志系统。只写你能重新运行并讲解的功能。

## 简历其他信息（必须本人填写）

学校：[学校名称]；专业：电子信息工程；学历：本科；毕业时间：2026 年 7 月。
联系方式：[手机]；邮箱：[邮箱]；城市意向：[填写]；教育/实践经历：[真实信息]。GitHub：`https://github.com/ryougireiko`。

## AI 辅助开发表述

项目允许合理使用 AI 辅助开发，但本人应承担设计、配置、运行、调试与质量验收，不能把未参与或不理解的成果表述成自己的独立研发经验。不要声称在真实企业落地上线。
