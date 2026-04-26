"""ファイルハッシュ計算。cleaning ログ / experiment ログで再現性担保用に使う。"""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK_SIZE = 1024 * 1024


def sha256_of_file(path: Path) -> str:
    """ファイル内容の SHA256 を `sha256:<hex>` 形式で返す。"""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK_SIZE):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"
