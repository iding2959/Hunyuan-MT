#!/bin/bash

# Hunyuan-MT OpenAI API Server 启动脚本
# 使用 vLLM 提供 OpenAI 兼容的 API 服务

# 设置变量
MODEL_PATH="/mnt/sdb/workspaces/project/hgmodel/Hunyuan-MT-Chimera-7B"
MODEL_NAME="Hunyuan-MT-Chimera-7B"
HOST="0.0.0.0"
PORT=8021
GPU_MEMORY_UTIL=0.92
TENSOR_PARALLEL_SIZE=1
DTYPE="bfloat16"
LOG_FILE="log_server.txt"

# 打印启动信息
echo "========================================="
echo "启动 Hunyuan-MT OpenAI API Server"
echo "========================================="
echo "模型路径: ${MODEL_PATH}"
echo "模型名称: ${MODEL_NAME}"
echo "监听地址: ${HOST}:${PORT}"
echo "GPU 内存利用率: ${GPU_MEMORY_UTIL}"
echo "张量并行大小: ${TENSOR_PARALLEL_SIZE}"
echo "数据类型: ${DTYPE}"
echo "日志文件: ${LOG_FILE}"
echo "========================================="
echo ""

# 启动服务器
python3 -m vllm.entrypoints.openai.api_server \
    --host ${HOST} \
    --port ${PORT} \
    --trust-remote-code \
    --model ${MODEL_PATH} \
    --served-model-name ${MODEL_NAME} \
    --gpu-memory-utilization ${GPU_MEMORY_UTIL} \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --dtype ${DTYPE} \
    --disable-log-stats \
    2>&1 | tee ${LOG_FILE}
