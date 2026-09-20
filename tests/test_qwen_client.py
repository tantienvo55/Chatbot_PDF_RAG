"""
Unit tests for QwenClient using mocked HTTP responses.
"""

import pytest
import httpx
from unittest.mock import MagicMock, patch

from src.generation.qwen_client import (
    QwenClient,
    QwenConnectionError,
    QwenTimeoutError,
    QwenAPIError,
)


def test_qwen_client_init_defaults() -> None:
    client = QwenClient()
    assert client.base_url == "http://localhost:11434"
    assert client.model_name == "qwen3:8b"
    assert client.temperature == 0.1
    assert client.timeout == 180.0


def test_qwen_client_custom_init() -> None:
    client = QwenClient(
        base_url="http://192.168.1.100:11434/",
        model_name="qwen-custom:latest",
        temperature=0.0,
        timeout=30.0,
    )
    assert client.base_url == "http://192.168.1.100:11434"
    assert client.model_name == "qwen-custom:latest"
    assert client.temperature == 0.0
    assert client.timeout == 30.0


def test_qwen_client_strips_thinking_tags() -> None:
    mock_http = MagicMock(spec=httpx.Client)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "message": {
            "role": "assistant",
            "content": "<think>\nThinking step 1: analyze law.\nThinking step 2: extract fine.\n</think>\nNgười điều khiển xe máy vi phạm nồng độ cồn bị phạt từ 2.000.000 đến 3.000.000 đồng."
        }
    }
    mock_http.post.return_value = mock_response

    client = QwenClient(client=mock_http)
    answer = client.generate("system prompt", "user question")

    assert "<think>" not in answer
    assert "</think>" not in answer
    assert "Người điều khiển xe máy vi phạm nồng độ cồn" in answer


def test_qwen_client_successful_generation() -> None:
    mock_http = MagicMock(spec=httpx.Client)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "message": {
            "role": "assistant",
            "content": "Theo Điểm c Khoản 4 Điều 11 Luật 36/2024/QH15, đèn đỏ mang ý nghĩa dừng lại."
        }
    }
    mock_http.post.return_value = mock_response

    client = QwenClient(client=mock_http)
    answer = client.generate("system prompt", "Đèn đỏ có được đi không?")

    assert "đèn đỏ mang ý nghĩa dừng lại" in answer
    mock_http.post.assert_called_once()
    call_args, call_kwargs = mock_http.post.call_args
    assert call_args[0] == "http://localhost:11434/api/chat"
    payload = call_kwargs["json"]
    assert payload["model"] == "qwen3:8b"
    assert payload["options"]["temperature"] == 0.1
    assert len(payload["messages"]) == 2


def test_qwen_client_connection_error() -> None:
    mock_http = MagicMock(spec=httpx.Client)
    mock_http.post.side_effect = httpx.ConnectError("Failed to connect")

    client = QwenClient(client=mock_http)
    with pytest.raises(QwenConnectionError, match="Cannot connect to Ollama"):
        client.generate("system", "user")


def test_qwen_client_timeout_error() -> None:
    mock_http = MagicMock(spec=httpx.Client)
    mock_http.post.side_effect = httpx.TimeoutException("Timed out")

    client = QwenClient(client=mock_http)
    with pytest.raises(QwenTimeoutError, match="timed out"):
        client.generate("system", "user")


def test_qwen_client_http_status_error() -> None:
    mock_http = MagicMock(spec=httpx.Client)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_http.post.side_effect = httpx.HTTPStatusError(
        "500 Internal Server Error",
        request=MagicMock(),
        response=mock_response,
    )

    client = QwenClient(client=mock_http)
    with pytest.raises(QwenAPIError, match="HTTP error 500"):
        client.generate("system", "user")


def test_qwen_client_malformed_response() -> None:
    mock_http = MagicMock(spec=httpx.Client)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"error": "unknown"}
    mock_http.post.return_value = mock_response

    client = QwenClient(client=mock_http)
    with pytest.raises(QwenAPIError, match="missing message.content"):
        client.generate("system", "user")
