# PDF Chat from Scratch

一个基于 LangChain + FAISS + 阿里云百炼的 PDF 对话应用。

## 功能

- 上传 PDF 文档
- 基于 RAG 的文档问答
- 支持多轮对话

## 技术栈

- **UI**: Streamlit
- **RAG 框架**: LangChain 1.x
- **向量库**: FAISS
- **模型**: 阿里云百炼 qwen-plus + text-embedding-v2

## 快速开始

### 1. 安装依赖

pip install -r requirements.txt

### 2. 配置 .env

OPENAI_API_KEY=sk-你的阿里云百炼Key
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1

### 3. 运行

streamlit run app.py