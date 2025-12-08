import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import Settings
from .logger import get_logger

logger = get_logger(__name__)


def _is_vllm_ready(base_url: str) -> bool:
    """探测 vLLM /v1/models 是否可用。"""
    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return False


def ensure_vllm_running(settings: Settings) -> None:
    """
    若 vLLM 未就绪则启动 start_server.sh，并阻塞等待 /v1/models 就绪或超时。
    """
    if _is_vllm_ready(settings.openai_base_url):
        logger.info("vLLM 已在运行，跳过启动。")
        return

    script_path = Path(__file__).resolve().parent.parent / settings.vllm_start_script
    if not script_path.exists():
        logger.error("未找到启动脚本: %s", script_path)
        raise FileNotFoundError(f"未找到启动脚本: {script_path}")

    logger.info("未检测到 vLLM，启动脚本: %s", script_path)
    subprocess.Popen(
        ["bash", str(script_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    deadline = time.time() + settings.vllm_ready_timeout
    while time.time() < deadline:
        if _is_vllm_ready(settings.openai_base_url):
            logger.info("vLLM 服务已就绪。")
            return
        time.sleep(settings.vllm_probe_interval)

    logger.error("vLLM 启动超时（%ss），请检查日志。", settings.vllm_ready_timeout)
    raise TimeoutError(f"vLLM 启动超时（{settings.vllm_ready_timeout}s），请检查日志。")
