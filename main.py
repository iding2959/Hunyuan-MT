import uvicorn

from services.config import get_settings
from services.vllm_launcher import ensure_vllm_running


def main() -> None:
    settings = get_settings()
    ensure_vllm_running(settings)
    uvicorn.run(
        "services.translation_service:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        reload_dirs=["services"],  # 仅监听服务代码，避免日志触发重载
        reload_excludes=["logs", "*.log", "logs/*"],
    )


if __name__ == "__main__":
    main()
