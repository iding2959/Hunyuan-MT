from openai import OpenAI

from .config import get_settings

_client: OpenAI | None = None


def get_openai_client() -> OpenAI:
    """懒加载 OpenAI 客户端。"""
    global _client
    if _client is None:
        settings = get_settings()
        _client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
    return _client
