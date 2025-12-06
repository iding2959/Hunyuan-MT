import vllm
import torch

# 检查 GPU 和 PyTorch 版本
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device:", torch.cuda.get_device_name(0))

# 检查是否安装 flash-attn
try:
    import flash_attn
    print("flash-attn installed:", flash_attn.__version__)
except ImportError:
    print("flash-attn not installed")

# 检查 FlashInfer 模块
try:
    import flash_infer
    print("FlashInfer is available:", flash_infer.__version__)
except ImportError:
    print("FlashInfer not installed or not found")

# 初始化 vLLM 引擎
from vllm import LLMEngine, SamplingParams

engine = LLMEngine(
    model="/mnt/sdb/workspaces/project/hgmodel/Hunyuan-MT-Chimera-7B",
    dtype=torch.bfloat16,
)

# 尝试生成一个小文本以触发 top-k/top-p 采样
prompt = "测试 FlashInfer 是否可用："

sampling_params = SamplingParams(
    temperature=0.7,
    top_k=20,
    top_p=0.6
)

result = engine.generate(prompt, sampling_params=sampling_params)
print("生成结果：", result[0].text)

# 结束
engine.shutdown()
