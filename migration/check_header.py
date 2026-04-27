"""CSV ファイルのヘッダ行と先頭 1 行だけを確認するユーティリティ（M2 限定）。

encoding を順に試し、最初に読めたものを採用する。
"""

from __future__ import annotations

import sys
from pathlib import Path

_ENCODINGS = ("utf-8-sig", "cp932", "utf-8")


def _looks_mojibake(text: str) -> bool:
    """Shift-JIS を utf-8 として読んだ際に出る latin-1 系制御域の混入で判定。"""
    if not text:
        return False
    suspicious = sum(1 for ch in text if 0x80 <= ord(ch) <= 0xFF)
    return suspicious / max(len(text), 1) > 0.1


def _read_first_two_lines(path: Path) -> tuple[str, str, str]:
    for enc in _ENCODINGS:
        try:
            with path.open("r", encoding=enc) as f:
                header = f.readline().rstrip("\n").rstrip("\r")
                first = f.readline().rstrip("\n").rstrip("\r")
        except UnicodeDecodeError:
            continue
        if enc == "utf-8-sig" and _looks_mojibake(header):
            continue
        return header, first, enc
    raise RuntimeError(f"No encoding in {_ENCODINGS} could decode {path}")


def main(directory: str) -> int:
    d = Path(directory)
    csvs = sorted(d.glob("*.csv"))
    if not csvs:
        print(f"No CSV files under {d}", file=sys.stderr)
        return 1
    sample = csvs[0]
    header, first, enc = _read_first_two_lines(sample)
    print(f"File     : {sample}")
    print(f"Encoding : {enc}")
    print(f"Columns  : {len(header.split(','))}")
    print(f"Header   : {header}")
    print(f"Row[0]   : {first}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
