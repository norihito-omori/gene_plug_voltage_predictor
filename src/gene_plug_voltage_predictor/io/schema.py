"""入力 CSV の期待スキーマ。M2 の実測ヘッダに合わせて列名を確定する。"""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, Field

# ADR-010: 機種間の列名フォーマット差異は io.csv_loader で正規化する。
# 正規化方向は EP370G（アンダースコア付き）を正とする。
CANONICAL_RENAMES: Final[dict[str, str]] = {
    "要求電圧1": "要求電圧_1",
    "要求電圧2": "要求電圧_2",
    "要求電圧3": "要求電圧_3",
    "要求電圧4": "要求電圧_4",
    "要求電圧5": "要求電圧_5",
    "要求電圧6": "要求電圧_6",
}


class InputSchema(BaseModel):
    """Phase 0 入力 CSV の期待カラム定義。

    specs/input_schema.md（M2 実測値）に基づく。列名は正規化後（ADR-010）の
    EP370G 形式を前提とする。
    """

    location_col: str = Field(default="target_id", description="機場ID 列")
    location_no_col: str = Field(
        default="target_no",
        description="機場No 列。ファイル名 {機場No}.csv と一致し、管理No_プラグNo の前半を構成",
    )
    mcnkind_col: str = Field(
        default="mcnkind_id",
        description="機種コード列。EP370G=54 / EP400G=115",
    )
    datetime_col: str = Field(
        default="dailygraphpt_ptdatetime",
        description="計測日時列（30 分粒度）",
    )
    rated_output_col: str = Field(
        default="target_output",
        description="定格出力(kW)列。EP370G=370 / EP400G=400",
    )
    voltage_cols: tuple[str, ...] = Field(
        default=(
            "要求電圧_1",
            "要求電圧_2",
            "要求電圧_3",
            "要求電圧_4",
            "要求電圧_5",
            "要求電圧_6",
        ),
        description="目的変数。CANONICAL_RENAMES 適用後の正規名（EP370G 形式）",
    )

    @property
    def required_columns(self) -> list[str]:
        return [
            self.location_col,
            self.location_no_col,
            self.mcnkind_col,
            self.datetime_col,
            self.rated_output_col,
            *self.voltage_cols,
        ]
