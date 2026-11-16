# 企业知识库智能助手｜RAG Knowledge Base V3

基于 **FastAPI + SQLAlchemy + MySQL + Redis + BGE Embedding + Chroma + BM25/RRF + DeepSeek + LangGraph + MCP + Streamlit** 的企业知识库问答与多工具任务助手。

> **项目定位**：个人工程实践与本地演示项目，使用虚构企业制度作为业务数据。项目展示文档入库、关系型数据持久化、混合检索、缓存、RAG、Agent 工具编排和容器化部署；不是已投产的多租户企业系统，请勿上传真实企业机密。

## 1. V3 升级内容

在 V2 的 RAG / Agent 能力上增加后端工程化能力：

- **MySQL + SQLAlchemy**：管理文档、Chunk 元数据；使用事务、索引、连接池和 JSON 字段。
- **Redis**：缓存检索结果，使用 TTL 控制过期；以数据库中的 `retrieval_revision` 作为缓存版本，文档或向量索引变更后旧缓存键自然失效。
- **故障降级**：Redis 不可用时跳过缓存，继续执行正常检索，不让缓存故障中断问答。
- **Docker Compose**：统一编排 FastAPI、Streamlit、MySQL、Redis；MySQL / Redis 配置健康检查和数据卷。
- **数据迁移**：提供 SQLite 文档元数据迁移到 MySQL 的脚本。
- **自动化测试**：本地测试使用 SQLite 和空缓存；CI 额外启动 MySQL、Redis 做真实集成测试。

## 2. 系统架构

```mermaid
flowchart TB
    U[用户] --> UI[Streamlit]
    UI --> API[FastAPI]
    API --> ORM[SQLAlchemy]
    ORM --> DB[(MySQL：docs / chunks / app_state)]
    API --> RET[Retriever]
    RET --> CACHE[(Redis：检索结果缓存)]
    RET --> BM25[BM25 关键词检索]
    RET --> DENSE[Chroma Dense 检索]
    BM25 --> RRF[RRF 融合与可选重排]
    DENSE --> RRF
    RRF --> RET
    RET --> LLM[DeepSeek RAG]
    API --> AGENT[LangGraph Agent]
    AGENT --> MCP[MCP 工具]
    AGENT --> CP[(SQLite Checkpoint)]
    LLM --> API
    API --> UI
```

### 组件职责

| 组件 | 职责 | 数据是否持久化 |
| --- | --- | --- |
| MySQL | 文档信息、Chunk 文本与元数据、检索版本号 | 是，独立数据卷 |
| Redis | 检索结果缓存，TTL 默认 300 秒 | AOF 持久化；缓存可重建 |
| Chroma | Dense 向量索引 | 是，位于模型数据卷 |
| SQLite | LangGraph Agent Checkpoint；本地测试向量后端 | 是，位于模型数据卷 |
| FastAPI | API、鉴权、业务服务装配 | 不直接存业务文件 |
| Streamlit | 上传、检索、问答和 Agent 操作界面 | 通过 API 访问后端 |

**设计边界**：MySQL 与 Chroma 是两个独立存储，不存在跨库原子事务。入库时先写元数据，再写向量；向量写入失败会补偿删除元数据。删除流程先清理向量，再删除 MySQL 记录。异常中断后可通过 `/api/admin/rebuild` 根据 MySQL 中的 Chunk 重新写入向量索引。该重建接口会重新 upsert 当前 Chunk，不等同于跨库事务或完整的数据修复系统。

## 3. 快速开始：本地轻量模式

本地默认使用 SQLite + Hash Embedding + Mock LLM，不要求安装或启动 MySQL、Redis，适合先跑测试和理解代码。

```bash
# Python 3.11 或 3.12
python -m venv .venv

# Windows PowerShell
.venv\\Scripts\\Activate.ps1

# macOS / Linux
# source .venv/bin/activate

pip install -r requirements-v3.txt
```

复制 `.env.example` 为 `.env`，设置一个本地 API Token。保持 `DEMO_MODE=true`，然后启动：

```bash
uvicorn enterprise.api:app --host 127.0.0.1 --port 8000
```

- API 文档：`http://127.0.0.1:8000/docs`
- 健康检查：`http://127.0.0.1:8000/health`

本地模式没有 Redis 时，缓存状态会显示 `disabled`；检索功能仍可使用。Hash Embedding 仅用于离线功能验证，不代表 BGE 语义检索效果。

## 4. Docker Compose：MySQL + Redis 模式

首次运行且项目中还没有 `.env` 时，复制配置模板：

```bash
cp .env.example .env
cp .env.example .env.api
```

如果你已经有 V2 的 `.env`（其中可能保存了 DeepSeek 密钥），不要直接覆盖它；保留原有密钥，只把模板中的 `MYSQL_ROOT_PASSWORD`、`MYSQL_DATABASE`、`MYSQL_USER`、`MYSQL_PASSWORD` 等缺失变量补充进去。

至少修改 `.env` 中的：

- `MYSQL_ROOT_PASSWORD`
- `MYSQL_PASSWORD`
- `APP_API_TOKEN`

在 Windows PowerShell 中可使用：

```powershell
Copy-Item .env.example .env
Copy-Item .env.example .env.api
```

启动四个服务：

```bash
docker compose -f docker-compose.v3.yml up -d --build
```

查看状态和日志：

```bash
docker compose -f docker-compose.v3.yml ps
docker compose -f docker-compose.v3.yml logs -f api
```

访问：

- Streamlit：`http://127.0.0.1:8501`
- FastAPI：`http://127.0.0.1:8000/docs`
- 健康检查：`http://127.0.0.1:8000/health`

Compose 会通过 `DB_HOST=mysql` 和 `REDIS_URL=redis://redis:6379/0` 连接容器网络中的服务。MySQL 和 Redis 不向宿主机发布端口；数据分别保存在 `mysql_data`、`redis_data` 和 `model_data` 卷中。

### 启用真实 BGE / Chroma / DeepSeek

如果需要完整 AI 能力，在 `.env.api` 中设置：

配置说明：`.env.api` 中的 `APP_API_TOKEN` 必须与 `.env` 中的值完全一致。复制模板生成 `.env.api` 后，请删除其中的 `MYSQL_ROOT_PASSWORD`，该密码只应保留在 `.env` 中。API 容器通过 `env_file: .env.api` 读取运行配置。

```dotenv
DEMO_MODE=false
DEEPSEEK_API_KEY=你的密钥
APP_API_TOKEN=与.env中完全相同的令牌
EMBEDDING_BACKEND=bge
VECTOR_BACKEND=chroma
LLM_BACKEND=deepseek
```

首次运行 BGE 会下载或加载模型缓存，镜像与模型体积较大。请确认磁盘空间充足。不要把 `.env`、`.env.api` 或其他真实密钥配置提交到 GitHub。

## 5. 从旧版 SQLite 迁移文档元数据到 MySQL

如果旧项目的 `APP_DATA_DIR` 中存在 `documents.sqlite3`，先启动 MySQL：

```bash
docker compose -f docker-compose.v3.yml up -d mysql redis
```

再运行迁移脚本：

```bash
docker compose -f docker-compose.v3.yml run --rm --no-deps api \\
  python scripts/migrate_sqlite_to_mysql.py --source /workspace/.local_data/documents.sqlite3
```

脚本按 SHA-256 跳过已经迁移的文档，可以重复执行。它迁移 `docs`、`chunks` 元数据，不复制 Chroma 或 SQLite 向量文件；Chunk ID 保持不变，原向量目录保留时可继续关联。若切换了向量后端或模型，请启动 API 后使用带鉴权的 `POST /api/admin/rebuild` 重建向量。

**迁移前请备份旧 `.local_data` 和 Docker volumes。不要在没有备份的情况下删除旧 SQLite 文件或旧数据卷。**

## 6. Redis 缓存设计

检索缓存位于 `enterprise/retrieval.py`，Redis 客户端封装在 `enterprise/cache.py`。

缓存键包含：

- 规范化后的查询文本；
- `top_k` 与检索策略；
- MySQL 中的 `retrieval_revision`；
- Embedding / Vector / Reranker 配置。

当文档新增、删除或向量索引重建时，数据库版本号递增。新查询使用新版本缓存键，旧键由 TTL 自动回收。这样不需要在文档更新时扫描和删除所有 Redis Key。

Redis 读取或写入失败会记录日志并降级到原始检索流程。Redis 是性能优化层，不是文档数据的唯一来源。

## 7. API 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 检查数据库和缓存状态 |
| GET | `/api/documents` | 文档列表 |
| POST | `/api/documents` | 上传 PDF / TXT / Markdown |
| DELETE | `/api/documents/{doc_id}` | 删除文档与向量 |
| POST | `/api/admin/rebuild` | 根据已登记 Chunk 重建向量索引 |
| POST | `/api/search` | BM25 / Dense / Hybrid 检索，使用 Redis 缓存 |
| POST | `/api/query` | RAG 问答 |
| POST | `/api/query/stream` | SSE 流式问答 |
| POST | `/api/agent` | LangGraph Agent 多工具任务 |

配置 `APP_API_TOKEN` 后，受保护接口使用 `Authorization: Bearer <token>`。

## 8. 测试

```bash
# 轻量测试：SQLite + Hash Embedding + Mock LLM + NullCache
python -m pytest -q tests_v2 tests_v3

# MySQL / Redis 集成测试：需要本机先启动服务并设置环境变量
# TEST_DATABASE_URL=mysql+pymysql://user:password@127.0.0.1:3306/enterprise_rag_test
# TEST_REDIS_URL=redis://127.0.0.1:6379/15
python -m pytest -q tests_integration
```

GitHub Actions 工作流会启动 MySQL 和 Redis 服务容器，并运行轻量测试及真实服务集成测试。当前开发环境若没有 Docker、PyMySQL、redis-py 或服务端实例，不能把本地测试结果当作真实 MySQL / Redis 联调结果。

## 9. 项目结构

```text
enterprise/
  api.py                 # FastAPI 路由、鉴权、健康检查
  cache.py               # Redis JSON 缓存与降级
  config.py              # 环境变量与数据库连接配置
  storage.py             # SQLAlchemy ORM、事务、数据版本
  service.py             # 服务装配与生命周期
  ingest.py              # 文档解析、切分、入库、索引同步
  retrieval.py           # BM25、Dense、RRF、Redis 缓存
  vector_index.py         # Chroma / SQLite 向量索引
  embedding.py            # BGE / Hash Embedding
  rag.py                  # RAG 证据与回答
  agent.py                # LangGraph Agent
  mcp_business.py         # MCP 业务工具
ui/
  app.py                  # Streamlit 界面
scripts/
  migrate_sqlite_to_mysql.py
  import_samples.py
  smoke_demo.py
tests_v2/                 # 原有 V2 功能回归测试
tests_v3/                 # SQLAlchemy 与 Redis 缓存测试
tests_integration/        # MySQL / Redis 真实服务集成测试
```

## 10. 已知限制

- 项目是单用户个人演示项目，不包含完整多租户、RBAC、审计与生产级密钥管理。
- MySQL 与 Chroma 不共享分布式事务；当前采用补偿操作和重建入口处理部分失败场景。
- Redis 缓存只覆盖检索结果，不缓存 LLM 最终回答。
- SQLite 向量后端用于小规模离线测试；大规模向量检索应使用 Chroma 或专用向量数据库。
- Hash Embedding 只用于确定性测试，不具备真正语义向量模型的效果。
- Agent Checkpoint 仍使用 SQLite，与 MySQL 文档元数据存储职责分离。
