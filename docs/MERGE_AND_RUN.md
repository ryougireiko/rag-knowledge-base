# 与原仓库合并 — Windows 操作说明（务必先看）

## 1. 为什么是补丁而不是重建仓库？

原仓库公开地址 `https://github.com/ryougireiko/rag-knowledge-base`，包含你在 Day11/12 手写的 `document.py`、`text_splitter.py`、`embeddings.py`、`retriever.py`、`main.py` 等。新的 `enterprise/` 代码会优先导入原 `document.Document` 与 `text_splitter.split_document`，保留你的 Git 提交历史。

生成环境无法直接 `git clone`，所以这里提供的是**完整新增模块压缩包**，而不是复制了原代码的镜像。补丁单独运行时有兼容切分器 fallback；合并到原库后才会使用你真实的 Day11 切分器。请保留原仓库文件并检查原测试。

## 2. 建立安全分支

```powershell
cd 'A:\桌面\day40\rag-knowledge-base'  # 替换成你的目录
 git status
 git switch -c feature/enterprise-agent
```

确认工作区无未提交变更，或者先备份/提交现有修改。**不要删除 `.git` 文件夹**。

## 3. 解压新增文件

把压缩包中 `enterprise/`、`ui/`、`tests_v2/`、`evaluation/`、`sample_company/`、`scripts/`、`docs/`、`.github/` 文件夹复制到旧仓库根目录；再复制 `requirements-v2.txt`、`.env.example`、`README_V2.md`、`Dockerfile.v2`、`docker-compose.v2.yml`、`.dockerignore.v2`、`.gitignore.v2-additions`。

若目录已有同名文件，先对比再合并。**不要覆盖**原 `document.py`、`loaders.py`、`text_splitter.py`、`embeddings.py`、`retriever.py`、`main.py`、`tests/`、`requirements.txt` 或 `README.md`。

把 `.gitignore.v2-additions` 内容追加到旧 `.gitignore`；将 `.dockerignore.v2` 的内容复制为 `.dockerignore`。

重要：原仓库有 `statistics.py`，这个文件名可能与 Python 标准库的 `statistics` 模块冲突。若安装依赖或运行报错中出现相关内容，应检查旧文件是否被错误导入，再通过 `git mv statistics.py chunk_statistics.py` 及修复对应 import 进行迁移，切勿先盲目覆盖。

## 4. Python 3.11 环境

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-v2.txt
Copy-Item .env.example .env
```

- 不要把 `.env` 或 `.local_data` 上传到 GitHub。
- 原先的 Anaconda Python 3.13 不建议直接复用，避免向量数据库、PyTorch 等依赖冲突。
- 如果 C 盘空间小，把 `.venv` 和 `APP_DATA_DIR` 都放在 A 盘项目下。
- `.env` Windows 路径推荐写成 `A:/AI_ProjectData`，避免 `\` 转义。

## 5. 第一轮：零成本离线验收

`.env`：`DEMO_MODE=true`。

```powershell
python -m pytest -q tests_v2
python -m pytest -q tests    # 原 Day11/12 测试也应全部通过
python -m scripts.smoke_demo
uvicorn enterprise.api:app --host 127.0.0.1 --port 8000
```

在 `http://127.0.0.1:8000/docs` 试 `/health`，上传样本文档，查询 `/api/query`，确认回答开头有「演示模式」。

如果你修改了 `.env` 后服务未重启，重启后再试。

## 6. 第二轮：真实 RAG 集成验收

修改 `.env`：

```ini
DEMO_MODE=false
DEEPSEEK_API_KEY=你自己的密钥
APP_API_TOKEN=至少32位随机字符串
RERANK_ENABLED=false
APP_DATA_DIR=A:/AI_ProjectData
MODEL_CACHE_DIR=A:/AI_ProjectData/models
```

重启 API；第一次文档导入会下载 BGE 模型、构建 Chroma。上传模拟文档并调用 `/api/query`。检查生成的答案是否有 `[S1]` 及来源，手动核对「普通员工在上海住宿 400 元/晚」与「3 晚 1200 元」。

可选启用 `RERANK_ENABLED=true`，会下载重排序模型；比较延迟与检索效果。

**注意**：演示模式建立的 HashEmbeddings 向量与真 BGE 向量维度不同。切换模式时**必须使用新的 `APP_DATA_DIR`**，不要混用向量索引。

## 7. 第三轮：Agent + MCP

保持真模式，调用：

```json
{"question":"查询公司制度，计算普通员工在上海出差3晚的住宿上限，标明文件来源。","thread_id":"my-demo-001"}
```

查看 `/api/agent` 的 `tool_trace` 是否包含 `search_company_knowledge` 和 `calculate_budget_via_mcp`。由于工具使用由 LLM 决定，如果第一次没有按预期选择工具，应收紧提示词或增加规则路由测试，而不是假称已经验证成功。

MCP 单独测试可运行 `python -m pytest -q tests_v2/test_mcp_business.py`。`python -m enterprise.mcp_business` 可开启 stdio MCP Server（该命令会持续等待 MCP Client 连接）。

## 8. 第四轮：Docker / Git

```powershell
docker compose -f docker-compose.v2.yml up --build
# 通过后，停止服务
 git status
 git diff --stat
 git add enterprise ui tests_v2 evaluation sample_company scripts docs .github README_V2.md requirements-v2.txt .env.example Dockerfile.v2 docker-compose.v2.yml .dockerignore .gitignore
 git status
 git commit -m "feat: add enterprise RAG and LangGraph agent v2"
 git push -u origin feature/enterprise-agent
```

上面 `git add` 命令使用的 `.dockerignore` 需要先从 `.dockerignore.v2` 复制生成；没有生成就把该项从命令中移除。首次提交前再次运行 `git status`，确认 `.env`、数据目录和模型权重没有被提交。

然后可在 GitHub 页面发起 Pull Request 到 `main`。先确认新旧测试均通过，再合并。最终再将 README_V2.md 经检查改为仓库的 README.md。

## 9. 遇到报错反馈方式

发给 ChatGPT：执行的命令、完整报错、Python 版本、`.env` 的**非敏感配置**和相关代码。**绝对不要把 API Key 或 Token 发出来。**
