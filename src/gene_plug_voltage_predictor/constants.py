"""機種別定数の初期版。対象機場リストは M2 の ADR-003 で確定後に追記。"""

from __future__ import annotations

from typing import Final

EP370G_RATED_POWER_KW: Final[int] = 370
EP400G_RATED_POWER_KW: Final[int] = 400

# M2 の ADR-003 で確定する。空リストのうちは `cli.clean` がエラー停止する想定。
EP370G_TARGET_LOCATIONS: Final[tuple[str, ...]] = ()
EP400G_TARGET_LOCATIONS: Final[tuple[str, ...]] = ()

SUPPORTED_MODEL_TYPES: Final[frozenset[str]] = frozenset({"EP370G", "EP400G"})
