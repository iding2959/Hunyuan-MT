#!/bin/bash

# =========================================
# Hunyuan-MT OpenAI API Server 高并发自动优化启动脚本（FP8 KV 缓存）
# 使用 vLLM 提供 OpenAI 兼容的 API 服务
# =========================================

# -------------------------
# 配置参数
# -------------------------
MODEL_PATH="/mnt/sdb/workspaces/project/hgmodel/Hunyuan-MT-Chimera-7B-fp8"
MODEL_NAME="Hunyuan-MT-Chimera-7B"
HOST="0.0.0.0"
PORT=8021
GPU_MEMORY_UTIL=0.92
TENSOR_PARALLEL_SIZE=1
DTYPE="bfloat16"
KV_CACHE_DTYPE="fp8"
LOG_FILE="log_server.txt"

# 单请求最小字符数（中文） → 转换为 token
MIN_CHARS=1000
TOKENS_PER_CHAR=1.6   # 约 1 token ≈ 0.6 中文字符
MAX_MODEL_LEN=$(python3 -c "print(int($MIN_CHARS * $TOKENS_PER_CHAR))")

# -------------------------
# 动态计算最大并发请求数
# -------------------------
# 获取 GPU 总显存 (MiB)
TOTAL_GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
TOTAL_GPU_MEM_GB=$(python3 -c "print($TOTAL_GPU_MEM/1024.0)")

# 估算每条请求 KV Cache 占用显存 (GB)
KV_MEM_PER_SEQ=$(python3 -c "print($MAX_MODEL_LEN * 7e-5)")  
# 注：7e-5 是经验系数，FP8 KV Cache 每 token 占用约 7e-5 GiB

# 计算最大并发数
MAX_NUM_SEQS=$(python3 -c "import math; print(max(1, math.floor($TOTAL_GPU_MEM_GB * $GPU_MEMORY_UTIL / $KV_MEM_PER_SEQ)))")

# -------------------------
# 修正 max_num_batched_tokens >= 模型最大长度
# -------------------------
MODEL_MAX_LEN=32768
MAX_NUM_BATCHED_TOKENS=$MODEL_MAX_LEN

# -------------------------
# 打印启动信息
# -------------------------
echo "========================================="
echo "启动 Hunyuan-MT OpenAI API Server（高并发 FP8 KV 缓存）"
echo "========================================="
echo "模型路径: ${MODEL_PATH}"
echo "模型名称: ${MODEL_NAME}"
echo "监听地址: ${HOST}:${PORT}"
echo "GPU 内存利用率: ${GPU_MEMORY_UTIL}"
echo "张量并行大小: ${TENSOR_PARALLEL_SIZE}"
echo "数据类型: ${DTYPE}"
echo "KV 缓存数据类型: ${KV_CACHE_DTYPE}"
echo "每条请求最大 token: ${MAX_MODEL_LEN}"
echo "单批次最大 token: ${MAX_NUM_BATCHED_TOKENS}"
echo "动态计算最大并发请求数: ${MAX_NUM_SEQS}"
echo "日志文件: ${LOG_FILE}"
echo "========================================="
echo ""

# -------------------------
# 启动服务器
# -------------------------
python3 -m vllm.entrypoints.openai.api_server \
    --host ${HOST} \
    --port ${PORT} \
    --trust-remote-code \
    --model ${MODEL_PATH} \
    --served-model-name ${MODEL_NAME} \
    --gpu-memory-utilization ${GPU_MEMORY_UTIL} \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --dtype ${DTYPE} \
    --kv-cache-dtype ${KV_CACHE_DTYPE} \
    --max-num-seqs ${MAX_NUM_SEQS} \
    --max-seq-len ${MAX_MODEL_LEN} \
    --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \
    --show-hidden-metrics-for-version latest \
    2>&1 | tee ${LOG_FILE}
