"""企业知识库智能助手 API：单进程演示项目，可选 Bearer Token 鉴权。
不包含企业级多租户身份认证与权限隔离。
"""
from __future__ import annotations
from contextlib import asynccontextmanager
import json
import logging
import re
import hmac
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Header, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from enterprise.service import Services

# 统一日志格式，输出到控制台
logging.basicConfig(level=logging.INFO,format='%(levelname)s %(name)s %(message)s')
log = logging.getLogger('enterprise.api')


class QueryRequest(BaseModel):
    """检索/问答请求体。"""
    query: str = Field(min_length=1, max_length=2000, description='用户问题或检索关键词')
    top_k: int = Field(default=5,ge=1,le=20, description='返回的候选文档片段数量（1～20）')
    strategy: str = Field(default='hybrid',pattern='^(hybrid|dense|bm25)$', description='检索策略：hybrid 混合检索、dense 语义检索、bm25 关键词检索')


class AgentRequest(BaseModel):
    """Agent 任务请求体。"""
    question: str = Field(min_length=1,max_length=2000, description='交给 Agent 处理的任务或问题')
    thread_id: str = Field(min_length=1,max_length=64,pattern=r'^[a-zA-Z0-9_-]+$', description='会话标识，用于关联多轮对话')


def create_app(service: Services | None = None):
    """应用工厂：构建并返回 FastAPI 实例。"""
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 启动时挂载 service（未传入则新建），关闭时释放资源
        app.state.service = service or Services() # 先传入service，如果是none等假值，就创建新的services()对象
        try: yield
        finally: app.state.service.close()

    app = FastAPI(
        title='企业知识库智能助手 API',
        description='单用户本地演示项目：支持文档管理、混合检索、RAG 问答及 Agent 工具调用。',
        version='2.0.0',
        lifespan=lifespan,
    )

    def get_svc(request: Request) -> Services:
        """依赖注入：从请求上下文取出服务实例。"""
        return request.app.state.service

    bearer_scheme = HTTPBearer(auto_error=False)

    def authorize(
            credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
            svc: Services = Depends(get_svc)
    ):
        """鉴权：配置了 api_token 时校验 Bearer Token，否则放行。"""
        token = svc.settings.api_token

        if token:
            # 使用 hmac.compare_digest 防止时序攻击
            if credentials is None or not hmac.compare_digest(
                    credentials.credentials,
                    token
            ):
                raise HTTPException(
                    status_code=401,
                    detail="Authorization required"
                )

    @app.get('/health', tags=['服务状态'], summary='检查服务运行状态')
    def health(svc: Services = Depends(get_svc)):
        """健康检查：返回当前后端配置。"""
        return {'status':'ok','mode':svc.settings.llm_backend,'vector_backend':svc.settings.vector_backend,
                'embeddings':svc.settings.embedding_backend}

    @app.get('/api/documents',dependencies=[Depends(authorize)], tags=['文档管理'], summary='获取已索引文档列表')
    def documents(svc: Services = Depends(get_svc)):
        """列出所有已索引文档。"""
        return {'documents':svc.store.docs()}

    @app.post('/api/documents',dependencies=[Depends(authorize)], tags=['文档管理'], summary='上传文档并建立索引', description='上传 PDF、TXT 或 Markdown 文档，完成解析、切分与索引。文件大小受后端配置限制。')
    async def upload(file: UploadFile = File(...),svc: Services=Depends(get_svc)):
        """上传文档：校验大小后解析并建立索引。"""
        if not file.filename: raise HTTPException(400,'Missing filename')
        # 先按限制读取，避免超大文件占用内存
        size = svc.settings.max_upload_mb * 1024 * 1024
        raw = await file.read(size+1)
        if len(raw)>size: raise HTTPException(413,'File is too large')
        try:
            result = svc.ingest.upload(file.filename,raw)
            log.info('uploaded document=%s chunks=%s',result['id'],result['chunks'])
            return result
        except ValueError as exc: raise HTTPException(400,str(exc)) from exc

    @app.delete('/api/documents/{doc_id}',dependencies=[Depends(authorize)], tags=['文档管理'], summary='删除指定文档及索引')
    def delete(doc_id: str,svc: Services=Depends(get_svc)):
        """删除文档：校验 id 格式并移除对应索引。"""
        if not re.fullmatch(r'[a-f0-9]{32}',doc_id): raise HTTPException(400,'invalid doc id')
        if not svc.ingest.delete(doc_id): raise HTTPException(404,'not found')
        return {'deleted': True}

    @app.post('/api/admin/rebuild',dependencies=[Depends(authorize)], tags=['文档管理'], summary='重建文档索引')
    def rebuild(svc: Services=Depends(get_svc)):
        """全量重建索引，返回索引后的片段数。"""
        return {'indexed_chunks':svc.ingest.rebuild()}

    @app.post('/api/search',dependencies=[Depends(authorize)], tags=['知识检索'], summary='检索相关文档片段')
    def search(request: QueryRequest,svc: Services=Depends(get_svc)):
        """纯检索：返回相关片段，不调用 LLM。"""
        return {'results':svc.retriever.search(request.query,request.top_k,request.strategy)}

    @app.post('/api/query',dependencies=[Depends(authorize)], tags=['RAG 问答'], summary='基于知识库生成回答')
    def query(request: QueryRequest,svc: Services=Depends(get_svc)):
        """RAG 问答：检索上下文后生成完整回答。"""
        try: return svc.rag.answer(request.query,request.top_k,request.strategy)
        except RuntimeError as exc: raise HTTPException(503,str(exc)) from exc

    @app.post('/api/query/stream',dependencies=[Depends(authorize)], tags=['RAG 问答'], summary='流式生成知识库回答', description='通过 SSE 依次返回来源、回答文本及完成事件。')
    def stream(request: QueryRequest,svc: Services=Depends(get_svc)):
        """流式问答：以 SSE 事件返回 sources/token/done。"""
        def events():
            try:
                # 先返回检索来源
                hits,sources=svc.rag.retrieve(request.query,request.top_k,request.strategy)
                yield f"event: sources\ndata: {json.dumps(sources,ensure_ascii=False)}\n\n"
                if not hits:
                    yield f"event: token\ndata: {json.dumps('知识库没有找到相关资料。',ensure_ascii=False)}\n\n"
                else:
                    # 逐块输出 LLM 生成的回答
                    for chunk in svc.llm.stream(request.query,svc.rag.context(hits)):
                        yield f"event: token\ndata: {json.dumps(chunk,ensure_ascii=False)}\n\n"
                yield 'event: done\ndata: {}\n\n'
            except Exception:
                log.exception('Streaming query failed')
                yield f"event: error\ndata: {json.dumps('生成失败，请检查服务器日志',ensure_ascii=False)}\n\n"
        return StreamingResponse(events(),media_type='text/event-stream',headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})

    @app.post('/api/agent',dependencies=[Depends(authorize)], tags=['Agent 工具调用'], summary='执行 Agent 多工具任务')
    async def agent(request: AgentRequest,svc: Services=Depends(get_svc)):
        """Agent 任务：延迟导入并执行多工具推理。"""
        from enterprise.agent import AgentService
        try: return await AgentService(svc).run(request.question,request.thread_id)
        except (ValueError,RuntimeError) as exc: raise HTTPException(503,str(exc)) from exc

    return app

# 模块级默认应用实例，供 uvicorn 直接加载
app = create_app()
