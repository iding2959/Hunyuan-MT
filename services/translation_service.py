from typing import Dict

from fastapi import FastAPI
from pydantic import BaseModel

from .config import get_settings
from .translator import translate_text
from .vllm_launcher import ensure_vllm_running

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
    translated_text = translate_text(payload.text)
    return TranslateResponse(text=translated_text)


if __name__ == "__main__":
    import uvicorn

    ensure_vllm_running(settings)
    uvicorn.run(
        "services.translation_service:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )
