from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gene_plug_voltage_predictor.io.csv_loader import load_raw_csv
from gene_plug_voltage_predictor.io.schema import InputSchema

_BASE_COLS = (
    "target_id,管理No,mcnkind_id,dailygraphpt_ptdatetime,target_output,"
    "発電機電力,累積運転時間,"
)
_EP370G_HEADER = _BASE_COLS + "要求電圧_1,要求電圧_2,要求電圧_3,要求電圧_4,要求電圧_5,要求電圧_6"
_EP370G_HEADER_WIDE = (
    _BASE_COLS
    + "要求電圧_1,要求電圧_2,要求電圧_3,要求電圧_4,要求電圧_5,要求電圧_6,"
    + "排気温度,機関回転数,その他1,その他2"  # 余分な 4 列
)
_EP400G_HEADER = _BASE_COLS + "要求電圧1,要求電圧2,要求電圧3,要求電圧4,要求電圧5,要求電圧6"


def _write_csv(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8-sig")


def test_load_raw_csv_rejects_file_with_wrong_mcnkind_id(tmp_path: Path) -> None:
    d = tmp_path / "EP370G_orig"
    d.mkdir()
    csv = d / "5630.csv"
    # data row says mcnkind_id=115 (EP400G code) while we expect 54 (EP370G)
    _write_csv(
        csv,
        _EP370G_HEADER + "\n"
        "140,5630,115,2018-03-27 00:30:00,370,300.0,1500.0,220,220,220,220,220,220\n",
    )
    with pytest.raises(ValueError, match="mcnkind_id mismatch"):
        load_raw_csv(
            csv,
            expected_mcnkind_id=54,
            expected_rated_kw=370,
            schema=InputSchema(),
        )


def test_load_raw_csv_accepts_matching_mcnkind_id(tmp_path: Path) -> None:
    d = tmp_path / "EP370G_orig"
    d.mkdir()
    csv = d / "5630.csv"
    _write_csv(
        csv,
        _EP370G_HEADER + "\n"
        "140,5630,54,2018-03-27 00:30:00,370,300.0,1500.0,220,221,222,223,224,225\n",
    )
    df = load_raw_csv(
        csv,
        expected_mcnkind_id=54,
        expected_rated_kw=370,
        schema=InputSchema(),
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df["mcnkind_id"].iloc[0] == 54
    assert df["target_output"].iloc[0] == 370


def test_load_raw_csv_rejects_mismatched_rated_output(tmp_path: Path) -> None:
    d = tmp_path / "EP370G_orig"
    d.mkdir()
    csv = d / "5630.csv"
    # target_output=400 but expected_rated_kw=370 → mismatch
    _write_csv(
        csv,
        _EP370G_HEADER + "\n"
        "140,5630,54,2018-03-27 00:30:00,400,300.0,1500.0,220,220,220,220,220,220\n",
    )
    with pytest.raises(ValueError, match="rated_output mismatch"):
        load_raw_csv(
            csv,
            expected_mcnkind_id=54,
            expected_rated_kw=370,
            schema=InputSchema(),
        )


def test_load_raw_csv_normalizes_ep400g_columns(tmp_path: Path) -> None:
    d = tmp_path / "EP400G_orig"
    d.mkdir()
    csv = d / "5760.csv"
    # EP400G raw has no-underscore variant; loader must rename to EP370G form.
    _write_csv(
        csv,
        _EP400G_HEADER + "\n"
        "159,5760,115,2018-03-27 00:30:00,400,300.0,1500.0,230,231,232,233,234,235\n",
    )
    df = load_raw_csv(
        csv,
        expected_mcnkind_id=115,
        expected_rated_kw=400,
        schema=InputSchema(),
    )
    # after normalization, canonical underscore names must exist
    for i in range(1, 7):
        assert f"要求電圧_{i}" in df.columns
    # and the non-underscore raw names must have been renamed away
    for i in range(1, 7):
        assert f"要求電圧{i}" not in df.columns
    assert df["要求電圧_1"].iloc[0] == 230
    assert df["要求電圧_6"].iloc[0] == 235


def test_load_raw_csv_accepts_nan_metadata_rows(tmp_path: Path) -> None:
    """NaN rows in mcnkind_id/target_output (new data format) are skipped by dropna()."""
    d = tmp_path / "EP370G_orig"
    d.mkdir()
    csv = d / "5630.csv"
    # Second row has NaN in mcnkind_id and target_output; non-NaN values are consistent.
    _write_csv(
        csv,
        _EP370G_HEADER + "\n"
        "140,5630,54,2018-03-27 00:30:00,370,300.0,1500.0,220,221,222,223,224,225\n"
        "140,5630,,2018-03-28 00:30:00,,310.0,1510.0,220,221,222,223,224,225\n",
    )
    df = load_raw_csv(
        csv,
        expected_mcnkind_id=54,
        expected_rated_kw=370,
        schema=InputSchema(),
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2


def test_load_raw_csv_keeps_only_required_columns(tmp_path: Path) -> None:
    """余分な列（排気温度等）はロード時に除外され、必要列のみ返る。"""
    d = tmp_path / "EP370G_orig"
    d.mkdir()
    csv = d / "5630.csv"
    _write_csv(
        csv,
        _EP370G_HEADER_WIDE + "\n"
        "140,5630,54,2018-03-27 00:30:00,370,"
        "300.0,1500.0,220,221,222,223,224,225,380.0,1500,foo,bar\n",
    )
    df = load_raw_csv(
        csv,
        expected_mcnkind_id=54,
        expected_rated_kw=370,
        schema=InputSchema(),
    )
    schema = InputSchema()
    # 必要列は全て存在する
    for col in schema.required_columns:
        assert col in df.columns, f"missing: {col}"
    # 余分な列は存在しない
    assert "排気温度" not in df.columns
    assert "機関回転数" not in df.columns
    assert "その他1" not in df.columns
