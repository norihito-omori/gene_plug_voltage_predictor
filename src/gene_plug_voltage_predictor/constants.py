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
    "9720": datetime(2021, 11, 1, 0, 0),
}

# ADR-003: Phase 0 対象外（要求電圧カラム欠損 / 稼働ゼロ / データ欠損）
EP370G_EXCLUDED_LOCATIONS: Final[tuple[str, ...]] = ("9520", "9570", "9571", "9850")

# EP400G: 要求電圧が異常値（スケール誤り）を示す期間の cutoff（以降のデータを使用）
# 目視確認に基づく各機場の異常終了日時
EP400G_START_DATETIMES: Final[Mapping[str, datetime]] = {
    "5760": datetime(2025, 2, 10, 7, 0),
    "5880": datetime(2024, 12, 27, 10, 30),
    "5881": datetime(2024, 12, 27, 10, 30),
    "8200": datetime(2021, 6, 12, 12, 0),
    "8930": datetime(2021, 6, 12, 20, 0),
    "8980": datetime(2021, 3, 29, 15, 0),
    "9040": datetime(2021, 6, 12, 13, 30),
    "9351": datetime(2018, 8, 1, 0, 0),
    "9470": datetime(2023, 4, 4, 9, 30),
    "9590": datetime(2020, 11, 29, 0, 0),
    "9630": datetime(2021, 11, 12, 0, 0),
    "9710": datetime(2021, 10, 17, 0, 0),
    "9800": datetime(2021, 9, 7, 0, 0),
}

# EP400G: 要求電圧が 2 固定（計測なし）またはファイル欠損で除外すべき機場
EP400G_EXCLUDED_LOCATIONS: Final[tuple[str, ...]] = (
    "6941",
    "7230",
    "7400",
    "7401",
    "7480",
    "7481",
    "7580",
    "7581",
    "7590",
    "7640",
    "7660",
    "7680",
    "7820",
    "7821",
    "7840",
    "7850",
    "7860",
    "7870",
    "7930",
    "7940",
    "8040",
    "8110",
    "8160",
    "8210",
    "8260",
    "8261",
    "8270",
    "8271",
    "8280",
    "8290",
    "8291",
    "8390",
    "8391",
    "8400",
    "8520",
    "8521",
    "8530",
    "8580",
    "8670",
    "8671",
    "8700",
    "8710",
    "8711",
    "8720",
    "8721",
    "8740",
    "8750",
    "8760",
    "8761",
    "8800",
    "8801",
    "8860",
    "8861",
    "8870",
    "8880",
    "8920",
    "8921",
    "8990",
    "8991",
    "9050",
)

# ADR-014 確定パラメータ
EXCHANGE_DETECTION_DEFAULTS: Final[dict[str, Any]] = {
    "voltage_drop_threshold": 5.0,
    "plug_quorum": 3,
    "window_days": 10,
    "min_days_each_side": 3,
    "merge_window_days": 7,
}
