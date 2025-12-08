import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from .config import get_settings

_configured = False


def _ensure_log_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def setup_logging() -> None:
    global _configured
    if _configured:
        return

    settings = get_settings()
    log_dir = Path(settings.log_dir)
    _ensure_log_dir(log_dir)

    log_file = log_dir / "api.log"

    handler = TimedRotatingFileHandler(
        filename=log_file,
        when="midnight",
        interval=1,
        backupCount=30,  # 保留 30 天
        encoding="utf-8",
        utc=False,
    )
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    root.addHandler(handler)

    # 控制台输出，便于开发调试
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
