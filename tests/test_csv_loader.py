from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gene_plug_voltage_predictor.io.csv_loader import load_raw_csv
from gene_plug_voltage_predictor.io.schema import InputSchema


def _write_csv(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8-sig")


def test_load_raw_csv_rejects_file_with_wrong_model_type(tmp_path: Path) -> None:
    d = tmp_path / "EP370G_orig"
    d.mkdir()
    csv = d / "5630.csv"
    _write_csv(
        csv,
        "location,model_type,measured_at,plug_voltage\n"
        "5630,EP400G,2024-04-01 00:00,220.0\n",
    )
    with pytest.raises(ValueError, match="model_type mismatch"):
        load_raw_csv(csv, expected_model_type="EP370G", schema=InputSchema())


def test_load_raw_csv_accepts_matching_model_type(tmp_path: Path) -> None:
    d = tmp_path / "EP370G_orig"
    d.mkdir()
    csv = d / "5630.csv"
    _write_csv(
        csv,
        "location,model_type,measured_at,plug_voltage\n"
        "5630,EP370G,2024-04-01 00:00,220.0\n",
    )
    df = load_raw_csv(csv, expected_model_type="EP370G", schema=InputSchema())
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df["model_type"].iloc[0] == "EP370G"
