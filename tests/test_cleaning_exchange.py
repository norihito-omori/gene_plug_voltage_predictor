from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gene_plug_voltage_predictor.cleaning.exchange import detect_exchange_events


def _synth_wide(
    *,
    start: str = "2023-01-01",
    days: int = 60,
    plug_means: tuple[float, ...] = (30.0, 30.0, 30.0, 30.0, 30.0, 30.0),
    exchanges: tuple[tuple[int, tuple[float, ...]], ...] = (),
    noise_std: float = 0.3,
    running_pattern: str = "always",
    rng_seed: int = 0,
) -> pd.DataFrame:
    """ADR-014 検出テスト用の合成横持ち DataFrame を返す。

    - 日時粒度: 30 分 1 レコード（運転中相当）
    - 列: dailygraphpt_ptdatetime, 発電機電力, 要求電圧_1..6
    - exchanges=[(day_offset, new_means), ...] で shift を差し込む
    - running_pattern='always' は全サンプル 発電機電力=300 (>0)
    """
    rng = np.random.default_rng(rng_seed)
    start_ts = pd.Timestamp(start)
    n = days * 48  # 30min × 48 = 1 day
    timestamps = pd.date_range(start=start_ts, periods=n, freq="30min")

    sorted_ev = sorted(exchanges, key=lambda x: x[0])

    def _means_at(day: int) -> np.ndarray:
        m = np.array(plug_means, dtype=float)
        for off, nm in sorted_ev:
            if day >= off:
                m = np.array(nm, dtype=float)
        return m

    voltages = np.zeros((n, 6), dtype=float)
    for i, ts in enumerate(timestamps):
        day = (ts - start_ts).days
        m = _means_at(day)
        voltages[i] = m + rng.normal(0, noise_std, size=6)

    power = np.full(n, 300.0) if running_pattern == "always" else np.zeros(n)

    return pd.DataFrame({
        "dailygraphpt_ptdatetime": timestamps,
        "発電機電力": power,
        "要求電圧_1": voltages[:, 0],
        "要求電圧_2": voltages[:, 1],
        "要求電圧_3": voltages[:, 2],
        "要求電圧_4": voltages[:, 3],
        "要求電圧_5": voltages[:, 4],
        "要求電圧_6": voltages[:, 5],
    })


_VCOLS = tuple(f"要求電圧_{i}" for i in range(1, 7))


def test_detects_clear_level_shift() -> None:
    df = _synth_wide(
        days=60,
        plug_means=(30, 30, 30, 30, 30, 30),
        exchanges=((30, (22, 22, 22, 22, 22, 22)),),
    )
    events = detect_exchange_events(df, voltage_cols=_VCOLS)
    assert len(events) == 1
    expected = pd.Timestamp("2023-01-31")  # start + 30 days
    assert abs((events[0] - expected).days) <= 3


def test_no_detection_without_shift() -> None:
    df = _synth_wide(days=60, exchanges=())
    events = detect_exchange_events(df, voltage_cols=_VCOLS)
    assert events == []


def test_min_days_each_side_gate() -> None:
    """交換の前後に運転日が 2 日しか無いと検出されない。"""
    df = _synth_wide(
        days=60,
        exchanges=((30, (22, 22, 22, 22, 22, 22)),),
    )
    ts = df["dailygraphpt_ptdatetime"]
    keep_days = {pd.Timestamp("2023-01-29"), pd.Timestamp("2023-01-30"),
                 pd.Timestamp("2023-02-01"), pd.Timestamp("2023-02-02")}
    mask = ts.dt.normalize().isin(keep_days)
    df.loc[~mask, "発電機電力"] = 0.0

    events = detect_exchange_events(
        df, voltage_cols=_VCOLS, min_days_each_side=3,
    )
    assert events == []


def test_nan_plug_excluded_from_quorum() -> None:
    """Plug 6 が全 NaN でも、残り 5 プラグで shift が検出されれば quorum=3 で検出。"""
    df = _synth_wide(
        days=60,
        exchanges=((30, (22, 22, 22, 22, 22, 22)),),
    )
    df["要求電圧_6"] = float("nan")
    events = detect_exchange_events(df, voltage_cols=_VCOLS)
    assert len(events) == 1


def test_merge_window_collapses_consecutive_days() -> None:
    """交換日周辺で複数日ヒットしても merge_window_days 以内なら 1 event にまとまる。"""
    df = _synth_wide(
        days=80,
        exchanges=((40, (22, 22, 22, 22, 22, 22)),),
    )
    events = detect_exchange_events(
        df, voltage_cols=_VCOLS, merge_window_days=7,
    )
    assert len(events) == 1


def test_cutoff_filter_drops_early_running() -> None:
    """cutoff より前の level shift は検出対象から除外される。"""
    df = _synth_wide(
        days=60,
        exchanges=((15, (22, 22, 22, 22, 22, 22)),),
    )
    events = detect_exchange_events(
        df, voltage_cols=_VCOLS, cutoff=pd.Timestamp("2023-02-01"),
    )
    assert events == []


def test_empty_input_returns_empty_list() -> None:
    """running 行がゼロの場合 ValueError ではなく [] を返す。"""
    df = _synth_wide(days=60, running_pattern="none")
    events = detect_exchange_events(df, voltage_cols=_VCOLS)
    assert events == []


def test_rejects_invalid_quorum() -> None:
    df = _synth_wide(days=10)
    with pytest.raises(ValueError, match="plug_quorum"):
        detect_exchange_events(df, voltage_cols=_VCOLS, plug_quorum=10)


def test_rejects_empty_voltage_cols() -> None:
    df = _synth_wide(days=10)
    with pytest.raises(ValueError, match="non-empty"):
        detect_exchange_events(df, voltage_cols=())
