import math
from anyio import to_thread

from fastapi import HTTPException
from openai import BadRequestError

from .config import get_settings
from .logger import get_logger
from .openai_client import get_openai_client

SYSTEM_PROMPT = "你是一名专业的翻译人员，请将给定的文本准确且自然地翻译成中文。"
CHUNK_TOKEN_LIMIT = 6200  # 期望单请求最大 token 上限
CHUNK_CHAR_LIMIT = CHUNK_TOKEN_LIMIT  # 直接使用 token 上限作为字符近似，避免额外估计
HARSH_CHAR_LIMIT = 6200  # 命中上下文超限错误时，按 1000 字符强制切分
logger = get_logger(__name__)


def _chunk_text(text: str, char_limit: int = CHUNK_CHAR_LIMIT) -> list[str]:
    """
    按字符数切分文本，确保每段不超过 char_limit。
    采用字符估算 token（约 4 字符 1 token），避免单次请求过大。
    优先按行聚合，过长行再切片。
    """
    if not text:
        return [""]

    chunks: list[str] = []
    current = ""

    for segment in text.splitlines(keepends=True):
        if len(segment) <= char_limit and len(current) + len(segment) <= char_limit:
            current += segment
            continue

        if current:
            chunks.append(current)
            current = ""

        if len(segment) <= char_limit:
            current = segment
        else:
            # 单行超长，硬切分
            for i in range(0, len(segment), char_limit):
                chunks.append(segment[i : i + char_limit])

    if current:
        chunks.append(current)

    return chunks or [""]


def _chunk_text_harsh(text: str, char_limit: int = HARSH_CHAR_LIMIT) -> list[str]:
    """强制按固定字符数切分（用于 BadRequest 上下文超限回退）。"""
    if not text:
        return [""]
    return [text[i : i + char_limit] for i in range(0, len(text), char_limit)]


def _translate_chunk(text: str) -> str:
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
        total_tokens = completion.usage.total_tokens if completion.usage else None
        logger.info("分片翻译完成，tokens=%s", total_tokens)
    except BadRequestError as exc:
        message = str(exc)
        if "maximum context length" in message or "context length" in message:
            logger.warning("命中上下文超限，按 %s 字符强制切分重试", HARSH_CHAR_LIMIT)
            sub_texts = _chunk_text_harsh(text, HARSH_CHAR_LIMIT)
            translated_parts = [_translate_chunk(sub) for sub in sub_texts]
            return "".join(translated_parts)
        logger.exception("调用翻译服务失败（BadRequest）")
        raise HTTPException(status_code=400, detail=f"调用翻译服务失败: {exc}") from exc
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


def translate_text(text: str) -> str:
    chunks = _chunk_text(text)
    if len(chunks) > 1:
        logger.info(
            "检测到长文本，分为 %s 段进行翻译（估算单段 <= %s token，约 %s 字符）",
            len(chunks),
            CHUNK_TOKEN_LIMIT,
            CHUNK_CHAR_LIMIT,
        )

    translated_parts = []
    for idx, chunk in enumerate(chunks, start=1):
        logger.info("翻译分片 %s/%s，长度=%s 字符", idx, len(chunks), len(chunk))
        translated_parts.append(_translate_chunk(chunk))

    return "".join(translated_parts)


async def translate_text_async(text: str) -> str:
    """
    异步包装，避免阻塞事件循环，将同步翻译逻辑丢到线程池。
    """
    return await to_thread.run_sync(translate_text, text, cancellable=True)
