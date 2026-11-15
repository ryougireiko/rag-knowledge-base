# Enterprise RAG V3 学习与交接指南

## 升级目标

在原有 Enterprise RAG V2 上增加 MySQL、SQLAlchemy、Redis 和多服务部署能力。升级过程中保留原有 API、RAG、Agent、MCP 与 Streamlit 功能。

## 学习顺序

### Stage 1：SQLAlchemy 与 MySQL

先看：

1. `enterprise/config.py`：`build_database_url()` 如何根据 `DATABASE_URL` 或 `DB_*` 环境变量生成连接串。
2. `enterprise/storage.py`：`DocumentRecord`、`ChunkRecord`、`AppStateRecord` 三张表分别负责什么。
3. `DocumentStore.add()`：为什么一个文档和它的 Chunk 必须放在同一个事务中。
4. `DocumentStore.delete()`：为什么删除时先清理 Chunk，再删除 Document。
5. `DocumentStore.revision()`：为什么检索缓存需要一个随数据变化而变化的版本号。

验收：不看代码，能够画出 docs、chunks、app_state 的关系，并解释一次上传的事务边界。

### Stage 2：Redis 缓存

先看：

1. `enterprise/cache.py`：Redis 客户端封装了哪些操作。
2. `enterprise/retrieval.py`：`_cache_key()` 为什么需要 query、top_k、strategy、revision 和模型配置。
3. `Retriever.search()`：缓存命中时为什么可以跳过 BM25 / Dense 检索。
4. Redis 不可用时为什么选择降级，而不是让整个问答请求失败。

验收：能够解释缓存命中、缓存未命中、TTL 到期、文档更新后旧缓存失效四种情况。

### Stage 3：入库一致性

先看 `enterprise/ingest.py`：

- 文档元数据写入 MySQL；
- Chunk 向量化并写入 Chroma；
- 向量写入失败时补偿删除 MySQL 文档；
- 文档删除时先清理向量，再删除关系数据；
- 重建索引后递增检索版本，避免旧缓存继续生效。

验收：能够说明 MySQL 和 Chroma 为什么不能依赖一个普通事务保证原子性，以及项目当前如何处理失败。

### Stage 4：Docker Compose 与联调

先看 `docker-compose.v3.yml`：

- API 如何通过 `DB_HOST=mysql` 访问 MySQL；
- API 如何通过 `REDIS_URL` 访问 Redis；
- `depends_on: condition: service_healthy` 的作用；
- MySQL、Redis、模型数据分别使用什么 volume；
- 为什么容器内访问服务使用服务名，而不是 `localhost`。

验收：能够独立启动、查看日志、进入容器、检查 MySQL 表和 Redis Key，并解释数据卷的用途。

## 面试表达边界

- 可以说项目使用 SQLAlchemy + MySQL 管理文档元数据，Redis 缓存检索结果。
- 可以说项目使用事务、连接池、TTL 和基于数据版本的缓存失效。
- 不要说成生产级高并发系统；项目没有经过生产流量压测。
- 不要编造 QPS、缓存命中率或响应时间。需要这些数据时，先设计可复现的压测并记录真实结果。
- MySQL / Redis 集成测试在真实服务环境通过后，才能在简历中写“已完成真实联调”。

## 当前需要实际验证的事情

1. 本地 `tests_v2` 通过。
2. GitHub Actions 的 MySQL / Redis 集成测试通过。
3. Docker Compose 启动四个服务成功。
4. `/health` 显示 database 为 `ok`、cache 为 `ok`。
5. 上传文档后 MySQL 中出现 docs / chunks 记录。
6. 重复查询能命中 Redis；上传或删除文档后新版本缓存键生效。
7. 迁移旧 SQLite 数据前先备份，并验证迁移数量。
