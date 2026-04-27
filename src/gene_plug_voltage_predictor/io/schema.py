"""入力 CSV の期待スキーマ。M2 の確認結果に基づいてカラム名を決定する。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class InputSchema(BaseModel):
    """Phase 0 入力 CSV の期待カラム定義。

    M2 の `check_header.py` 結果に応じて差し替える。
    """

    location_col: str = Field(default="location", description="機場 No 列")
    model_type_col: str = Field(default="model_type", description="機種列")
    datetime_col: str = Field(default="measured_at", description="計測日時列")
    target_col: str = Field(default="plug_voltage", description="目的変数（プラグ電圧）")

    @property
    def required_columns(self) -> list[str]:
        return [self.location_col, self.model_type_col, self.datetime_col, self.target_col]
