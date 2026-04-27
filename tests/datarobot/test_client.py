"""client.py のテスト."""

from __future__ import annotations

import pytest
from pytest_mock import MockerFixture

from gene_plug_voltage_predictor.datarobot import client as client_mod


def test_setup_client_raises_when_token_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(EnvironmentError, match="DATAROBOT_API_TOKEN"):
        client_mod.setup_client()


def test_setup_client_uses_token_from_env(
    monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    monkeypatch.setenv("DATAROBOT_API_TOKEN", "tok-abc")
    dr_client = mocker.patch.object(client_mod.dr, "Client")

    client_mod.setup_client()

    dr_client.assert_called_once_with(
        token="tok-abc",
        endpoint="https://app.datarobot.com/api/v2",
    )


def test_setup_client_uses_custom_endpoint(
    monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    monkeypatch.setenv("DATAROBOT_API_TOKEN", "tok-abc")
    monkeypatch.setenv("DATAROBOT_ENDPOINT", "https://custom.example.com/api/v2")
    dr_client = mocker.patch.object(client_mod.dr, "Client")

    client_mod.setup_client()

    dr_client.assert_called_once_with(
        token="tok-abc",
        endpoint="https://custom.example.com/api/v2",
    )
