"""機種別定数の初期版。対象機場リストは M2 の ADR-003 で確定後に追記。"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any, Final

EP370G_RATED_POWER_KW: Final[int] = 370
EP400G_RATED_POWER_KW: Final[int] = 400

# M2 の ADR-003 で確定する。空リストのうちは `cli.clean` がエラー停止する想定。
EP370G_TARGET_LOCATIONS: Final[tuple[str, ...]] = ()
EP400G_TARGET_LOCATIONS: Final[tuple[str, ...]] = ()

SUPPORTED_MODEL_TYPES: Final[frozenset[str]] = frozenset({"EP370G", "EP400G"})

# ADR-012 per-location cutoff（本番データに混じる早期・異常期間を切り捨てる開始日時）
# amend 履歴: 9610/9611（2026-04-28）, 9580/9581（2026-04-28 第2波）
EP370G_START_DATETIMES: Final[Mapping[str, datetime]] = {
    "5630": datetime(2022, 6, 26, 0, 30),
    "8950": datetime(2021, 6, 12, 20, 30),
    "9290": datetime(2022, 7, 21, 10, 0),
    "9380": datetime(2023, 5, 31, 0, 30),
    "9381": datetime(2023, 5, 31, 14, 0),
    "9580": datetime(2023, 7, 21, 0, 30),
    "9581": datetime(2023, 7, 21, 0, 30),
    "9610": datetime(2024, 4, 2, 19, 30),
    "9611": datetime(2024, 4, 2, 20, 0),
    "9690": datetime(2024, 3, 19, 8, 30),
}

# ADR-003: Phase 0 対象外（要求電圧カラム欠損 / 稼働ゼロ）
EP370G_EXCLUDED_LOCATIONS: Final[tuple[str, ...]] = ("9570", "9571", "9850")

# ADR-014 確定パラメータ
EXCHANGE_DETECTION_DEFAULTS: Final[dict[str, Any]] = {
    "voltage_drop_threshold": 5.0,
    "plug_quorum": 3,
    "window_days": 10,
    "min_days_each_side": 3,
    "merge_window_days": 7,
}
