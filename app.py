"""
Chat with PDF：Streamlit + LangChain RAG（FAISS）+ DashScope 兼容 OpenAI 接口。
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 读取 .env 中的 DashScope 兼容 OpenAI 配置
load_dotenv()

# ---------------------------------------------------------------------------
# 模型与检索链
# ---------------------------------------------------------------------------


def get_embeddings() -> OpenAIEmbeddings:
    """DashScope 文本向量：text-embedding-v2。"""
    return OpenAIEmbeddings(
        model="text-embedding-v2",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
        # 百炼模型不走 OpenAI tokenizer，关闭上下文长度校验避免报错
        check_embedding_ctx_length=False,
    )


def get_llm() -> ChatOpenAI:
    """对话模型：qwen-plus，走 DashScope 兼容模式 base_url。"""
    return ChatOpenAI(
        model="qwen-plus",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
        temperature=0.2,
    )


def build_rag_chain(vectorstore: FAISS, llm: ChatOpenAI):
    """用检索到的 PDF 片段回答问题。

    优先 create_retrieval_chain + create_stuff_documents_chain；
    LangChain 1.x 若无 classic 包，则用等价 LCEL 组合。
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是 PDF 文档问答助手。请只根据下面的上下文回答用户问题；"
                "若上下文没有相关信息，请明确说明无法从文档中找到答案，不要编造。\n\n"
                "上下文：\n{context}",
            ),
            ("human", "{input}"),
        ]
    )

    try:
        try:
            from langchain_classic.chains import create_retrieval_chain
            from langchain_classic.chains.combine_documents import (
                create_stuff_documents_chain,
            )
        except ImportError:
            from langchain.chains import create_retrieval_chain
            from langchain.chains.combine_documents import create_stuff_documents_chain

        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        return create_retrieval_chain(retriever, combine_docs_chain)
    except ImportError:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnableLambda, RunnablePassthrough

        def format_docs(docs: list) -> str:
            return "\n\n".join(doc.page_content for doc in docs)

        # LCEL：检索 -> 填入 prompt -> 生成 -> 统一成 {"answer": ...}
        return (
            RunnablePassthrough.assign(
                context=lambda x: format_docs(retriever.invoke(x["input"]))
            )
            | prompt
            | llm
            | StrOutputParser()
            | RunnableLambda(lambda text: {"answer": text})
        )


def ask_rag(rag_chain: Any, question: str) -> str:
    """调用问答链，取出生成的回答文本。"""
    result = rag_chain.invoke({"input": question})
    if isinstance(result, dict):
        return result.get("answer") or result.get("output") or str(result)
    return str(result)


# ---------------------------------------------------------------------------
# PDF -> 切分 -> FAISS
# ---------------------------------------------------------------------------


def pdf_to_chunks(uploaded_file) -> list:
    """把 Streamlit 上传的 PDF 解析并切分成文本块。"""
    # PyPDFLoader 需要本地路径，先把内存中的文件写到临时目录
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
    finally:
        os.remove(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return splitter.split_documents(documents)


def ingest_pdf(uploaded_file, embeddings: OpenAIEmbeddings) -> None:
    """解析 PDF，写入（或追加到）session_state 中的 FAISS 向量库。"""
    chunks = pdf_to_chunks(uploaded_file)
    if not chunks:
        raise ValueError("未能从 PDF 中提取到文本，请检查文件是否为可复制文本的 PDF。")

    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
    else:
        st.session_state.vectorstore.add_documents(chunks)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------


def init_session_state() -> None:
    """对话历史、向量库、已入库文件名都放在 session_state 里跨 rerun 保留。"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "indexed_files" not in st.session_state:
        # {文件标识: 显示名}，避免同一文件被重复向量化
        st.session_state.indexed_files = {}


def render_sidebar(embeddings: OpenAIEmbeddings) -> None:
    """侧边栏：上传 PDF，入库后提示「已加入知识库」。"""
    with st.sidebar:
        st.header("知识库")
        uploaded_file = st.file_uploader(
            "上传 PDF",
            type=["pdf"],
            help="上传后会自动解析、切分并写入 FAISS。",
        )

        if uploaded_file is not None:
            file_id = f"{uploaded_file.name}-{uploaded_file.size}"
            if file_id not in st.session_state.indexed_files:
                with st.spinner("正在解析 PDF 并构建向量库..."):
                    try:
                        ingest_pdf(uploaded_file, embeddings)
                        st.session_state.indexed_files[file_id] = uploaded_file.name
                    except Exception as exc:
                        st.error(f"构建知识库失败：{exc}")

        if st.session_state.indexed_files:
            st.success("已加入知识库")
            for name in st.session_state.indexed_files.values():
                st.caption(f"• {name}")
        else:
            st.info("请先上传 PDF，再开始提问。")


def render_chat(llm: ChatOpenAI) -> None:
    """主区域：展示历史，用 chat_input 提问。"""
    st.title("Chat with PDF")
    st.caption("基于 LangChain RAG + FAISS，模型为阿里云百炼 qwen-plus。")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("向这份 PDF 提问…")
    if not question:
        return

    if st.session_state.vectorstore is None:
        st.warning("请先在左侧上传 PDF。")
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("正在检索相关片段并生成回答..."):
            try:
                rag_chain = build_rag_chain(st.session_state.vectorstore, llm)
                answer = ask_rag(rag_chain, question)
            except Exception as exc:
                answer = f"生成回答失败：{exc}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


def main() -> None:
    st.set_page_config(page_title="Chat with PDF", page_icon="📄", layout="wide")

    api_key = os.getenv("OPENAI_API_KEY", "")
    api_base = os.getenv("OPENAI_API_BASE", "")
    if not api_key or "你的阿里云" in api_key or not api_base:
        st.error("请先在 `.env` 中填写有效的 OPENAI_API_KEY 和 OPENAI_API_BASE。")
        st.stop()

    init_session_state()
    embeddings = get_embeddings()
    llm = get_llm()
    render_sidebar(embeddings)
    render_chat(llm)


if __name__ == "__main__":
    main()
