from fastapi import HTTPException

from .config import get_settings
from .logger import get_logger
from .openai_client import get_openai_client

SYSTEM_PROMPT = "你是一名专业的翻译人员，请将给定的文本准确且自然地翻译成中文。"
logger = get_logger(__name__)


def translate_text(text: str) -> str:
    client = get_openai_client()
    settings = get_settings()

    try:
        completion = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        )
        logger.info("翻译完成，返回 tokens=%s", completion.usage and completion.usage.total_tokens)
    except Exception as exc:  # pragma: no cover - 网络调用错误兜底
        logger.exception("调用翻译服务失败")
        raise HTTPException(status_code=500, detail=f"调用翻译服务失败: {exc}") from exc

    try:
        translated_text = completion.choices[0].message.content
    except Exception as exc:  # pragma: no cover - 防御性解析
        logger.exception("解析响应失败")
        raise HTTPException(status_code=500, detail=f"解析响应失败: {exc}") from exc

    if translated_text is None:
        logger.error("未获取到翻译结果")
        raise HTTPException(status_code=500, detail="未获取到翻译结果")

    return translated_text
