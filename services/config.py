import os
from functools import lru_cache

from pydantic import BaseModel


def load_env(filepath: str) -> None:
    """加载简单的 key=value 环境变量文件。"""
    if not os.path.isfile(filepath):
        return
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key and value and key not in os.environ:
                os.environ[key.strip()] = value.strip()


# 先尝试 .env，不存在则 fallback 到 env.example
load_env(".env")
load_env("env.example")


def _str_to_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return str(value).lower() in {"1", "true", "t", "yes", "y", "on"}


class Settings(BaseModel):
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "http://localhost:8021/v1")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "EMPTY")
    openai_model: str = os.getenv("OPENAI_MODEL", "Hunyuan-MT-Chimera-7B")

    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8030"))
    api_reload: bool = _str_to_bool(os.getenv("API_RELOAD"), True)

    vllm_start_script: str = os.getenv("VLLM_START_SCRIPT", "start_server.sh")
    vllm_ready_timeout: int = int(os.getenv("VLLM_READY_TIMEOUT", "120"))
    vllm_probe_interval: float = float(os.getenv("VLLM_PROBE_INTERVAL", "2"))

    log_dir: str = os.getenv("LOG_DIR", "logs")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
