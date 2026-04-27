"""pytest 共通 fixture."""

from __future__ import annotations

from pathlib import Path

import pytest


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    """テスト fixture ディレクトリのパス."""
    return FIXTURES_DIR


@pytest.fixture(autouse=True)
def _clear_datarobot_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """各テストの前に DATAROBOT_* 環境変数をクリア."""
    monkeypatch.delenv("DATAROBOT_API_TOKEN", raising=False)
    monkeypatch.delenv("DATAROBOT_ENDPOINT", raising=False)
