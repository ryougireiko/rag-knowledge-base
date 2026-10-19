"""Streamlit UI; API token stays server-side in UI process environment."""
import os
import json
import uuid
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

# 定位项目根目录，不依赖终端当前工作目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 加载项目根目录下的 .env
load_dotenv(PROJECT_ROOT / ".env")

# FastAPI 服务地址
API = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000"
).rstrip("/")

# 从环境变量读取 API Token
TOKEN = os.getenv("APP_API_TOKEN", "")

# 统一请求头
HEADERS = {
    "Authorization": f"Bearer {TOKEN}"
} if TOKEN else {}

# 只输出是否读取成功，不泄露 Token
print(f"[Streamlit] API Token loaded: {bool(TOKEN)}")
st.set_page_config(page_title='企业知识库智能助手',page_icon='📚',layout='wide')
st.title('📚 企业知识库智能助手')
st.caption('企业知识库 · 混合检索 RAG · LangGraph Agent（本地演示项目，请勿上传真实企业机密）')

if 'thread_id' not in st.session_state:
    st.session_state.thread_id=uuid.uuid4().hex
if 'conversation' not in st.session_state:
    st.session_state.conversation=[]

with st.sidebar:
    st.subheader('知识库文件')
    file=st.file_uploader('选择文档（PDF / TXT / Markdown）',type=['pdf','txt','md','markdown'])
    if file and st.button('上传并索引',use_container_width=True):
        try:
            r=requests.post(API+'/api/documents',headers=HEADERS,
                files={'file':(file.name,file.getvalue())},timeout=180)
            r.raise_for_status(); st.success('文档上传并索引成功')
        except Exception as exc: st.error(str(exc))
    if st.button('刷新文档列表',use_container_width=True):
        try:
            r=requests.get(API+'/api/documents',headers=HEADERS,timeout=15)
            r.raise_for_status(); st.session_state.docs=r.json()['documents']
        except Exception as exc: st.error(str(exc))
    for d in st.session_state.get('docs',[]):
        st.write(f"{d['filename']} （{d['chunks']} 个文本块）")
        if st.button('删除 '+d['id'][:8],key=d['id']):
            try:
                r=requests.delete(API+'/api/documents/'+d['id'],headers=HEADERS,timeout=15)
                r.raise_for_status(); st.success('已删除；请刷新列表')
            except Exception as exc: st.error(str(exc))
    if st.button('新建 Agent 会话'):
        st.session_state.thread_id=uuid.uuid4().hex
        st.session_state.conversation=[]
    st.caption('单用户本地演示项目，不包含企业级权限隔离。上传大小以服务端配置为准。')

mode=st.radio('模式',options=['RAG 问答','Agent 多工具任务'],horizontal=True)
strategy=st.selectbox(
    '检索方式',
    ['hybrid', 'dense', 'bm25'],
    format_func=lambda value: {
        'hybrid': '混合检索（推荐）',
        'dense': '语义检索',
        'bm25': '关键词检索',
    }[value],
    disabled=(mode != 'RAG 问答'),
)
use_stream=st.checkbox('实时流式显示回答',value=True,disabled=(mode!='RAG 问答'))
for role,content in st.session_state.conversation:
    with st.chat_message(role): st.markdown(content)
question=st.chat_input('例如：出差三晚的住宿报销上限如何计算？')
if question:
    st.session_state.conversation.append(('user',question))
    with st.chat_message('user'): st.markdown(question)
    with st.chat_message('assistant'):
        try:
            if mode=='RAG 问答':
                payload={'query':question,'top_k':5,'strategy':strategy}
                if use_stream:
                    response=requests.post(API+'/api/query/stream',headers=HEADERS,
                        json=payload,stream=True,timeout=180)
                    response.raise_for_status()
                    sources=[]
                    def tokens():
                        nonlocal_sources=sources
                        event_name=''
                        for line in response.iter_lines(decode_unicode=True):
                            if line is None: continue
                            if isinstance(line,bytes): line=line.decode('utf-8')
                            if line.startswith('event: '): event_name=line[7:]
                            elif line.startswith('data: '):
                                body=json.loads(line[6:])
                                if event_name=='sources': nonlocal_sources.extend(body)
                                elif event_name=='token': yield body
                                elif event_name=='error': yield '[服务端错误] '+str(body)
                    answer=st.write_stream(tokens())
                    response.close()
                    data={'sources':sources}
                else:
                    r=requests.post(API+'/api/query',headers=HEADERS,json=payload,timeout=180)
                    r.raise_for_status(); data=r.json()
                    answer=data['answer']; st.markdown(answer)
                if data.get('sources'):
                    with st.expander('检索来源'):
                        for s in data['sources']:
                            st.write(f"[{s['source_id']}] {s['filename']} · 文本块 {s['chunk_index']} · {s['preview']}")
            else:
                r=requests.post(API+'/api/agent',headers=HEADERS,
                    json={'question':question,'thread_id':st.session_state.thread_id},timeout=180)
                r.raise_for_status(); data=r.json()
                answer=data['answer']; st.markdown(answer)
                with st.expander('工具调用轨迹'):
                    for trace in data.get('tool_trace',[]): st.write(trace)
            st.session_state.conversation.append(('assistant',answer))
        except Exception as exc: st.error(f'请求失败：{exc}')
