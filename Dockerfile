# ============================================================
# Dockerfile - AI CRM 智能客户经营平台
# ============================================================
# 基于 Python 3.10  slim 镜像，轻量安全
# 使用通义千问 text-embedding-v2 在线 API，零模型下载
# 重排序模型（Cross-Encoder ~80MB）在首次使用时懒加载
#
# 构建：
#   docker build -t ai-crm .
#
# 运行：
#   docker run -d -p 8501:8501 --env-file .env ai-crm
#
# 或使用 docker-compose（推荐）：
#   docker-compose up -d
# ============================================================

FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # HuggingFace 镜像（国内用户如无 VPN 请取消注释）
    # HF_ENDPOINT=https://hf-mirror.com \
    # 默认使用 Streamlit 端口
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# 安装系统依赖（chromadb 和 sentence-transformers 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖清单
COPY requirements.txt ./
# 可选的 pip 源：国内构建时传 --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ARG PIP_INDEX_URL=https://pypi.org/simple
# 可选的 CPU 版 torch 源：本项目只用 CPU 推理（RERANKER_DEVICE=cpu），
# 先装 CPU 轮子可把 torch 体积从 ~800MB（CUDA 版）降到 ~200MB。
# 例：--build-arg TORCH_FIND_LINKS=https://mirrors.aliyun.com/pytorch-wheels/cpu/
ARG TORCH_FIND_LINKS=
# pip cache 挂载：BuildKit 会保留已下载的 wheel，构建中断后重试可续传，
# 不会把几个 GB 的依赖（torch 等）从头再下一遍。
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    if [ -n "$TORCH_FIND_LINKS" ]; then \
        pip install --find-links "$TORCH_FIND_LINKS" torch; \
    fi && \
    pip install --index-url ${PIP_INDEX_URL} -r requirements.txt

# 复制项目源码（AI_CRM/ 内已包含 .env.example 与 sql/mysql 建表脚本）
COPY AI_CRM/ ./AI_CRM/

# 创建数据持久化目录
RUN mkdir -p AI_CRM/utils/chroma_kb AI_CRM/logfile

# 暴露端口
EXPOSE 8501 8000

# 健康检查：打 Streamlit 自带探针（不会触发页面脚本/后端初始化）
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# 默认启动 Streamlit 前端
CMD ["streamlit", "run", "AI_CRM/app.py"]
