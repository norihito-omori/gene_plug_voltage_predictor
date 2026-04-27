"""DataRobot API クライアント初期化."""

from __future__ import annotations

import logging
import os

import datarobot as dr

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "https://app.datarobot.com/api/v2"


def setup_client() -> None:
    """DataRobot API クライアントを初期化する.

    OS 環境変数 DATAROBOT_API_TOKEN を必須とする.
    DATAROBOT_ENDPOINT は任意（デフォルト: https://app.datarobot.com/api/v2）.

    Raises:
        EnvironmentError: DATAROBOT_API_TOKEN が未設定の場合.
    """
    token = os.environ.get("DATAROBOT_API_TOKEN")
    if not token:
        raise EnvironmentError(
            "DATAROBOT_API_TOKEN is not set. "
            "Set it as an OS environment variable before running."
        )
    endpoint = os.environ.get("DATAROBOT_ENDPOINT") or DEFAULT_ENDPOINT
    logger.info("Initializing DataRobot client (endpoint=%s)", endpoint)
    dr.Client(token=token, endpoint=endpoint)
