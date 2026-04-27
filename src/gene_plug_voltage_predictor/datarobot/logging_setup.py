"""JST タイムゾーン対応の logging 初期化ユーティリティ."""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

JST = ZoneInfo("Asia/Tokyo")
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def _jst_converter(timestamp: float) -> time.struct_time:
    """logging.Formatter.converter 用の JST タイムコンバータ."""
    return datetime.fromtimestamp(timestamp, tz=JST).timetuple()


def setup_logging(
    log_file: Path | None = None,
    level: int = logging.INFO,
) -> None:
    """ルートロガーを JST で初期化する.

    Args:
        log_file: 指定時はファイルにも書き出す（デフォルト有効で呼び出す想定）.
        level: ログレベル. デフォルトは INFO.
    """
    formatter = logging.Formatter(LOG_FORMAT)
    formatter.converter = _jst_converter

    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        root.addHandler(fh)
