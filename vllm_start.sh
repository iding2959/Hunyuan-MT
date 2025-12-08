#!/bin/bash

# =========================================
# Hunyuan-MT OpenAI API Server 启动脚本（16精度）
# 使用 vLLM 提供 OpenAI 兼容的 API 服务
# =========================================

MODEL_PATH="/mnt/sdb/workspaces/project/hgmodel/Hunyuan-MT-Chimera-7B"
MODEL_NAME="Hunyuan-MT-Chimera-7B"
HOST="0.0.0.0"
PORT=8021
GPU_MEMORY_UTIL=0.92
TENSOR_PARALLEL_SIZE=1
DTYPE="bfloat16"
LOG_DIR="logs"

MAX_MODEL_LEN=1550
MAX_NUM_BATCHED_TOKENS=32768

# -------------------------
# 自动计算最大并发请求数
# -------------------------
TOTAL_GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
TOTAL_GPU_MEM_GB=$(awk "BEGIN{printf \"%.2f\", $TOTAL_GPU_MEM/1024}")

MODEL_MEM_GB=8
ACTIVATION_MEM_GB=5

AVAILABLE_MEM_GB=$(awk "BEGIN{printf \"%.2f\", $TOTAL_GPU_MEM_GB*$GPU_MEMORY_UTIL - $MODEL_MEM_GB - $ACTIVATION_MEM_GB}")

# 每条请求大约占用显存 (GiB)
# bfloat16 = 2 bytes/token
PER_SEQ_MEM_GB=$(awk "BEGIN{printf \"%.2f\", $MAX_MODEL_LEN*2/1024/1024/1024}")

if (( $(awk "BEGIN{print ($PER_SEQ_MEM_GB <= 0)}") )); then
    PER_SEQ_MEM_GB=0.01
fi

MAX_NUM_SEQS=$(awk "BEGIN{printf \"%d\", $AVAILABLE_MEM_GB/$PER_SEQ_MEM_GB}")

if (( MAX_NUM_SEQS > 512 )); then
    MAX_NUM_SEQS=512
fi

# -------------------------
# 打印信息
# -------------------------
echo "========================================="
echo "启动 Hunyuan-MT OpenAI API Server（16精度 bfloat16）"
echo "========================================="
echo "模型路径: ${MODEL_PATH}"
echo "模型名称: ${MODEL_NAME}"
echo "监听地址: ${HOST}:${PORT}"
echo "GPU 内存利用率: ${GPU_MEMORY_UTIL}"
echo "张量并行大小: ${TENSOR_PARALLEL_SIZE}"
echo "数据类型: ${DTYPE}"
echo "每条请求最大 token: ${MAX_MODEL_LEN}"
echo "单批次最大 token: ${MAX_NUM_BATCHED_TOKENS}"
echo "动态计算最大并发请求数: ${MAX_NUM_SEQS}"
echo "日志文件目录: ${LOG_DIR}"
echo "========================================="
echo ""

# -------------------------
# 日志轮转设置（按日期命名）
# -------------------------
# 创建日志目录（如果不存在的话）
mkdir -p $LOG_DIR

# 获取当前日期（格式：YYYY-MM-DD）
LOG_DATE=$(date +%Y-%m-%d)
LOG_FILE="${LOG_DIR}/log_server_${LOG_DATE}.txt"

# 保留最近的 10 个日志文件（按日期命名）
LOG_ROTATE_COUNT=2

# 删除最旧的日志文件（如果日志文件超过 10 个，并且是以 log_server_ 开头的）
LOG_FILES=$(ls $LOG_DIR/vllm_server_* 2>/dev/null)  # 只列出以 log_server_ 开头的文件
LOG_COUNT=$(echo "$LOG_FILES" | wc -w)  # 计算文件数量

if [ $LOG_COUNT -ge $LOG_ROTATE_COUNT ]; then
    OLDEST_LOG=$(echo "$LOG_FILES" | sort | head -n 1)  # 按时间排序，获取最旧的日志文件
    rm "${OLDEST_LOG}"  # 删除最旧的日志文件
fi

# -------------------------
# 启动服务器并记录日志
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
    --max-num-seqs ${MAX_NUM_SEQS} \
    --max-seq-len ${MAX_MODEL_LEN} \
    --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \
    --show-hidden-metrics-for-version latest \
    2>&1 | tee ${LOG_FILE}   # 将日志记录到以日期命名的日志文件中
