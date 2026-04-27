"""クリーニング CLI: raw CSV 群 -> 学習用 CSV + クリーニングログ Markdown ドラフト。"""

from __future__ import annotations

import argparse
import sys
from datetime import date as _date
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from gene_plug_voltage_predictor.cleaning.pipeline import (
    CleaningPipeline,
    StepSpec,
)
from gene_plug_voltage_predictor.cleaning.reporters import render_cleaning_log
from gene_plug_voltage_predictor.io.csv_loader import load_raw_csv
from gene_plug_voltage_predictor.io.schema import InputSchema
from gene_plug_voltage_predictor.utils.hashing import sha256_of_file


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


def main() -> int:
    ap = argparse.ArgumentParser(description="Clean raw CSVs into training dataset")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cleaning-log", required=True, type=Path)
    ap.add_argument("--author", default="大森")
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    model_type: str = cfg["model_type"]
    input_dir = Path(cfg["input_dir"])
    target_locations: list[int | str] = cfg.get("target_locations") or []
    if not target_locations:
        print(
            f"ERROR: target_locations is empty in {args.config}. "
            "Fill in after ADR-003 is accepted.",
            file=sys.stderr,
        )
        return 2

    schema = InputSchema()
    frames: list[pd.DataFrame] = []
    for loc in target_locations:
        csv = input_dir / f"{loc}.csv"
        if not csv.exists():
            print(f"ERROR: CSV not found for location {loc}: {csv}", file=sys.stderr)
            return 2
        frames.append(
            load_raw_csv(csv, expected_model_type=model_type, schema=schema)
        )
    df = pd.concat(frames, ignore_index=True)

    specs = _build_specs(cfg.get("steps") or [])
    pipeline = CleaningPipeline(specs)
    cleaned, history = pipeline.run(df)

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
