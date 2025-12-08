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
    )


if __name__ == "__main__":
    main()
