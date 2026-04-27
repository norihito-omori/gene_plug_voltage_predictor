"""クリーニング CLI: raw CSV 群 -> 学習用 CSV + クリーニングログ Markdown ドラフト。

ADR-014 対応: 機場ごとに交換検出を実行し、events を pipeline の
assign_generation に runtime_params 経由で注入する。--events-out で
検出結果を副次 CSV として書き出せる。
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date as _date
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from gene_plug_voltage_predictor.cleaning.exchange import detect_exchange_events
from gene_plug_voltage_predictor.cleaning.pipeline import (
    CleaningPipeline,
    StepSpec,
)
from gene_plug_voltage_predictor.cleaning.reporters import render_cleaning_log
from gene_plug_voltage_predictor.constants import (
    EP370G_EXCLUDED_LOCATIONS,
    EP370G_START_DATETIMES,
    EXCHANGE_DETECTION_DEFAULTS,
)
from gene_plug_voltage_predictor.io.csv_loader import load_raw_csv
from gene_plug_voltage_predictor.io.schema import InputSchema
from gene_plug_voltage_predictor.utils.hashing import sha256_of_file

_logger = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return loaded


def _build_specs(raw_steps: list[dict[str, Any]]) -> list[StepSpec]:
    specs: list[StepSpec] = []
    for s in raw_steps:
        if not s.get("enabled", True):
            continue
        specs.append(
            StepSpec(
                name=s["name"],
                adr=s["adr"],
                params=s.get("params", {}),
            )
        )
    return specs


def _write_events_csv(
    events_by_location: dict[str, list[pd.Timestamp]], out_path: Path
) -> None:
    rows: list[dict[str, Any]] = []
    for loc, evs in events_by_location.items():
        for ev in evs:
            rows.append({"target_no": loc, "exchange_date": ev.date().isoformat()})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["target_no", "exchange_date"]).to_csv(
        out_path, index=False, encoding="utf-8-sig"
    )


_REQUIRED_CONFIG_KEYS = (
    "model_type",
    "expected_mcnkind_id",
    "rated_power_kw",
    "input_dir",
    "dataset_name",
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Clean raw CSVs into training dataset")
    ap.add_argument(
        "--config", required=True, type=Path,
        help="クリーニング設定 YAML (model_type, target_locations, steps など)",
    )
    ap.add_argument(
        "--out", required=True, type=Path,
        help="クリーニング済み CSV の出力先",
    )
    ap.add_argument(
        "--cleaning-log", required=True, type=Path,
        help="クリーニングログ Markdown の出力先",
    )
    ap.add_argument(
        "--events-out", type=Path, default=None,
        help="交換検出イベントの CSV 出力先(省略時は出力しない)",
    )
    ap.add_argument(
        "--author", default="大森",
        help="cleaning ログの著者名 (default: 大森)",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = _load_yaml(args.config)
    missing = [k for k in _REQUIRED_CONFIG_KEYS if k not in cfg]
    if missing:
        print(
            f"ERROR: required config keys missing in {args.config}: {missing} "
            "(see specs/input_schema.md §4 / ADR-010).",
            file=sys.stderr,
        )
        return 2
    model_type: str = cfg["model_type"]
    expected_mcnkind_id: int = int(cfg["expected_mcnkind_id"])
    expected_rated_kw: int = int(cfg["rated_power_kw"])
    input_dir = Path(cfg["input_dir"])
    target_locations: list[int | str] = cfg.get("target_locations") or []
    if not target_locations:
        print(
            f"ERROR: target_locations is empty in {args.config}. "
            "Fill in after ADR-003 is accepted.",
            file=sys.stderr,
        )
        return 2

    # ADR-003: excluded との重複を拒否(EP370G のみチェック)
    if model_type == "EP370G":
        dup = sorted(set(str(x) for x in target_locations) & set(EP370G_EXCLUDED_LOCATIONS))
        if dup:
            print(
                f"ERROR: target_locations includes EXCLUDED_LOCATIONS: {dup}",
                file=sys.stderr,
            )
            return 2

    schema = InputSchema()
    frames: list[pd.DataFrame] = []
    events_by_location: dict[str, list[pd.Timestamp]] = {}
    cutoff_map = EP370G_START_DATETIMES if model_type == "EP370G" else {}

    for loc in target_locations:
        loc_str = str(loc)
        csv = input_dir / f"{loc_str}.csv"
        if not csv.exists():
            print(f"ERROR: CSV not found for location {loc_str}: {csv}", file=sys.stderr)
            return 2
        df_loc = load_raw_csv(
            csv,
            expected_mcnkind_id=expected_mcnkind_id,
            expected_rated_kw=expected_rated_kw,
            schema=schema,
        )
        frames.append(df_loc)

        cutoff_dt = cutoff_map.get(loc_str)
        cutoff_ts = pd.Timestamp(cutoff_dt) if cutoff_dt is not None else None
        if cutoff_ts is None:
            _logger.info("%s: no cutoff (all rows considered)", loc_str)
        try:
            events_by_location[loc_str] = detect_exchange_events(
                df_loc,
                voltage_cols=schema.voltage_cols,
                cutoff=cutoff_ts,
                **EXCHANGE_DETECTION_DEFAULTS,
            )
        except ValueError as e:
            print(f"ERROR: {loc_str}.csv: {e}", file=sys.stderr)
            return 2
        _logger.info(
            "%s: detected %d exchange events",
            loc_str, len(events_by_location[loc_str]),
        )

    if args.events_out:
        _write_events_csv(events_by_location, args.events_out)
        print(f"[OK] wrote events CSV to {args.events_out}")

    df = pd.concat(frames, ignore_index=True)
    specs = _build_specs(cfg.get("steps") or [])
    pipeline = CleaningPipeline(specs)
    cleaned, history = pipeline.run(
        df,
        runtime_params={
            "assign_generation": {"events_by_location": events_by_location},
        },
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(args.out, index=False, encoding="utf-8-sig")
    out_hash = sha256_of_file(args.out)

    md = render_cleaning_log(
        history=history,
        dataset_name=cfg["dataset_name"],
        model_type=model_type,
        author=args.author,
        run_date=_date.today(),
        output_path=str(args.out).replace("\\", "/"),
        output_hash=out_hash,
        related_adrs=[
            "decision-001",
            "decision-002",
            "decision-003",
            "decision-009",
            "decision-010",
            "decision-012",
            "decision-014",
        ]
        + [s.adr for s in specs],
    )
    args.cleaning_log.parent.mkdir(parents=True, exist_ok=True)
    args.cleaning_log.write_text(md, encoding="utf-8")

    print(f"[OK] wrote {args.out} ({len(cleaned)} rows, {out_hash})")
    print(f"[OK] wrote cleaning log draft to {args.cleaning_log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
