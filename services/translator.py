from fastapi import HTTPException

from .config import get_settings
from .openai_client import get_openai_client

SYSTEM_PROMPT = "你是一名专业的翻译人员，请将给定的文本准确且自然地翻译成中文。"


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
    except Exception as exc:  # pragma: no cover - 网络调用错误兜底
        raise HTTPException(status_code=500, detail=f"调用翻译服务失败: {exc}") from exc

    try:
        translated_text = completion.choices[0].message.content
    except Exception as exc:  # pragma: no cover - 防御性解析
        raise HTTPException(status_code=500, detail=f"解析响应失败: {exc}") from exc

    if translated_text is None:
        raise HTTPException(status_code=500, detail="未获取到翻译结果")

    return translated_text
