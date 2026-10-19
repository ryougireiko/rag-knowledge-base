# 生成时验证记录（2026-10-17）

## 本轮实际执行

- `python -m compileall -q enterprise ui tests_v2 evaluation scripts` —— 已通过。
- `DEMO_MODE=true python -m pytest -q tests_v2` —— 8 passed，1 skipped。跳过项为 MCP Python SDK 未安装，不能当作测试通过。
- `DEMO_MODE=true python -m scripts.smoke_demo` —— 通过，导入 `travel_policy.md`，`/RAG` 服务返回 `[S1]` 来源与演示文本。**并未调用真实 LLM。**
- 导入 `sample_company` 8 个示例文件后执行检索评估（仅 HashEmbeddings/SQLite 的非语义演示环境）：23 个问题，dense Hit@5=1.000，BM25 Hit@5=1.000，hybrid Hit@5=1.000；因为只有 8 个文件且 K=5，**没有统计外推意义，更不能用于简历宣传或性能改进结论。**

## 未在当前环境实测

- BGE 模型下载/加载，Chroma 实库兼容性。
- DeepSeek 真 API 回答与 token SSE。
- LangGraph SQLite Checkpoint 多轮 Agent 调用。
- MCP v2 Client/Server 真实工具调用。
- Streamlit 与真后端的联调。
- Docker Compose 镜像构建和启动。
- Reranker、真实评测数据、性能与安全压测。

**交付状态定义：可合并的工程源码候选 + 经测试的离线核心链路；不是已完全验收的求职展示系统。** 用户在本机完成未测项验收、修复后，再将实际结果填入 README/简历。
