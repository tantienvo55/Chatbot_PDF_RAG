"""
Local Qwen client module using Ollama HTTP API.
"""

import os
import re
from typing import Any, Optional
import httpx


DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_QWEN_MODEL = "qwen3:8b"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TIMEOUT_SECONDS = 300.0


class QwenError(Exception):
    """Base exception for Qwen client errors."""
    pass


class QwenConnectionError(QwenError):
    """Raised when unable to connect to the local Ollama instance."""
    pass


class QwenTimeoutError(QwenError):
    """Raised when Ollama request times out."""
    pass


class QwenAPIError(QwenError):
    """Raised when Ollama API returns an error response or malformed payload."""
    pass


class QwenClient:
    """
    Client for interacting with local Qwen models hosted on Ollama.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        client: Optional[httpx.Client] = None,
    ) -> None:
        """
        Initialize QwenClient.

        Args:
            base_url: Ollama base URL (default from OLLAMA_BASE_URL or http://localhost:11434).
            model_name: Qwen model tag (default from QWEN_MODEL or qwen3:8b).
            temperature: Sampling temperature (default: 0.1 for deterministic legal QA).
            timeout: HTTP timeout in seconds (default: 120.0s).
            client: Optional pre-configured httpx.Client for testing/mocking.
        """
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE_URL).rstrip("/")
        self.model_name = model_name or os.environ.get("QWEN_MODEL") or DEFAULT_QWEN_MODEL
        self.temperature = float(temperature)
        self.timeout = float(timeout)
        self._custom_client = client
        self.last_metrics: dict[str, Any] = {}

    def _get_client(self) -> httpx.Client:
        if self._custom_client is not None:
            return self._custom_client
        return httpx.Client(timeout=self.timeout)

    def clean_thinking_tags(self, text: str) -> str:
        """Remove any internal thinking/reasoning tags emitted by reasoning models."""
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return cleaned.strip()

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send chat completion request to local Ollama.

        Args:
            system_prompt: Grounded system instruction.
            user_prompt: User question with retrieved context.
            temperature: Optional override for temperature.

        Returns:
            Cleaned response string.
        """
        url = f"{self.base_url}/api/chat"
        temp = self.temperature if temperature is None else float(temperature)

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": temp,
            },
        }

        client = self._get_client()
        try:
            res = client.post(url, json=payload)
            res.raise_for_status()
            data = res.json()
            metric_keys = [
                "total_duration",
                "load_duration",
                "prompt_eval_count",
                "prompt_eval_duration",
                "eval_count",
                "eval_duration",
            ]
            self.last_metrics = {k: data[k] for k in metric_keys if k in data}
        except httpx.ConnectError as e:
            raise QwenConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. Please ensure Ollama is running."
            ) from e
        except httpx.TimeoutException as e:
            raise QwenTimeoutError(
                f"Request to Ollama at {self.base_url} timed out after {self.timeout}s."
            ) from e
        except httpx.HTTPStatusError as e:
            raise QwenAPIError(
                f"Ollama API returned HTTP error {e.response.status_code}: {e.response.text}"
            ) from e
        except Exception as e:
            raise QwenAPIError(f"Unexpected error communicating with Ollama: {e}") from e
        finally:
            if self._custom_client is None:
                client.close()

        msg = data.get("message", {})
        content = msg.get("content")
        if content is None:
            raise QwenAPIError(f"Malformed response from Ollama: missing message.content in {data}")

        return self.clean_thinking_tags(content)
