from typing import Dict

from fastapi import FastAPI
from pydantic import BaseModel

from .config import get_settings
from .logger import get_logger, setup_logging
from .translator import translate_text
from .vllm_launcher import ensure_vllm_running

setup_logging()
logger = get_logger(__name__)

app = FastAPI(title="Hunyuan-MT Translation Service")
settings = get_settings()


class TranslateRequest(BaseModel):
    text: str


class TranslateResponse(BaseModel):
    text: str


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/translate", response_model=TranslateResponse)
async def translate(payload: TranslateRequest) -> TranslateResponse:
    """翻译接口：仅接收待翻译文本，返回译文。"""
    logger.info("收到翻译请求，文本长度=%s", len(payload.text or ""))
    translated_text = translate_text(payload.text)
    return TranslateResponse(text=translated_text)


if __name__ == "__main__":
    import uvicorn

    logger.info("启动 API 服务，host=%s port=%s reload=%s", settings.api_host, settings.api_port, settings.api_reload)
    ensure_vllm_running(settings)
    uvicorn.run(
        "services.translation_service:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        reload_dirs=["services"],  # 仅监控服务代码，避免日志更新触发
        reload_excludes=["logs", "*.log", "logs/*"],
    )
