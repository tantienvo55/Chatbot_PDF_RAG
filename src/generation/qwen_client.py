"""
Local Qwen client module using Ollama HTTP API.
Supports both synchronous completion (generate) and token streaming (generate_stream)
with cross-chunk think-tag suppression, TTFT tracking, num_predict control, and model keep-alive.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Generator, Optional
import httpx


DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_QWEN_MODEL = "qwen3:8b"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TIMEOUT_SECONDS = 180.0
DEFAULT_NUM_PREDICT = 384
DEFAULT_KEEP_ALIVE = "10m"


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


class StreamThinkFilter:
    """
    Statefully filters <think>...</think> blocks across streaming token chunks.
    Ensures reasoning tags and inner chain-of-thought are never exposed to the user.
    """

    def __init__(self) -> None:
        self.in_think: bool = False
        self.buffer: str = ""

    def feed(self, chunk: str) -> str:
        """
        Process an incoming text chunk and return safe, cleaned user-visible text.
        """
        if not chunk:
            return ""

        self.buffer += chunk
        output_parts: list[str] = []

        while self.buffer:
            if not self.in_think:
                start_idx = self.buffer.find("<think>")
                if start_idx != -1:
                    # Text before <think> is safe to emit
                    output_parts.append(self.buffer[:start_idx])
                    self.in_think = True
                    self.buffer = self.buffer[start_idx + len("<think>"):]
                else:
                    # Check if buffer ends with a partial prefix of "<think>"
                    tag = "<think>"
                    partial_len = 0
                    for i in range(1, min(len(self.buffer), len(tag)) + 1):
                        if tag.startswith(self.buffer[-i:]):
                            partial_len = i
                    if partial_len > 0:
                        safe_text = self.buffer[:-partial_len]
                        output_parts.append(safe_text)
                        self.buffer = self.buffer[-partial_len:]
                        break
                    else:
                        output_parts.append(self.buffer)
                        self.buffer = ""
                        break
            else:
                end_idx = self.buffer.find("</think>")
                if end_idx != -1:
                    # Found closing tag: exit thinking mode
                    self.in_think = False
                    self.buffer = self.buffer[end_idx + len("</think>"):].lstrip()
                else:
                    # Still inside think block; check if buffer ends with partial prefix of "</think>"
                    close_tag = "</think>"
                    partial_len = 0
                    for i in range(1, min(len(self.buffer), len(close_tag)) + 1):
                        if close_tag.startswith(self.buffer[-i:]):
                            partial_len = i
                    if partial_len > 0:
                        self.buffer = self.buffer[-partial_len:]
                    else:
                        self.buffer = ""
                    break

        return "".join(output_parts)

    def finalize(self) -> str:
        """Flush any remaining buffered text when stream ends."""
        if not self.in_think and self.buffer:
            res = self.buffer
            self.buffer = ""
            return res
        return ""


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
        num_predict: Optional[int] = None,
        keep_alive: Optional[str] = None,
        client: Optional[httpx.Client] = None,
    ) -> None:
        """
        Initialize QwenClient.

        Args:
            base_url: Ollama base URL (default from OLLAMA_BASE_URL or http://localhost:11434).
            model_name: Qwen model tag (default from QWEN_MODEL or qwen3:8b).
            temperature: Sampling temperature (default: 0.1 for deterministic legal QA).
            timeout: HTTP timeout in seconds (default: 180.0s).
            num_predict: Maximum tokens to predict (default from QWEN_NUM_PREDICT or 384).
            keep_alive: Ollama memory keep-alive duration (default from OLLAMA_KEEP_ALIVE or '10m').
            client: Optional pre-configured httpx.Client for testing/mocking.
        """
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE_URL).rstrip("/")
        self.model_name = model_name or os.environ.get("QWEN_MODEL") or DEFAULT_QWEN_MODEL
        self.temperature = float(temperature)
        self.timeout = float(timeout)

        # num_predict configuration
        if num_predict is not None:
            self.num_predict: Optional[int] = int(num_predict)
        elif "QWEN_NUM_PREDICT" in os.environ:
            self.num_predict = int(os.environ["QWEN_NUM_PREDICT"])
        else:
            self.num_predict = DEFAULT_NUM_PREDICT

        # keep_alive configuration
        self.keep_alive: Optional[str] = (
            keep_alive
            if keep_alive is not None
            else os.environ.get("OLLAMA_KEEP_ALIVE", DEFAULT_KEEP_ALIVE)
        )
        self.think = False

        self._custom_client = client
        self.last_metrics: dict[str, Any] = {}
        self.last_ttft_ms: Optional[float] = None

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
        num_predict: Optional[int] = None,
    ) -> str:
        """
        Send synchronous chat completion request to local Ollama.

        Args:
            system_prompt: Grounded system instruction.
            user_prompt: User question with retrieved context.
            temperature: Optional override for temperature.
            num_predict: Optional override for max tokens.

        Returns:
            Cleaned response string.
        """
        url = f"{self.base_url}/api/chat"
        temp = self.temperature if temperature is None else float(temperature)
        eff_predict = num_predict if num_predict is not None else self.num_predict

        options_dict: dict[str, Any] = {"temperature": temp}
        if eff_predict is not None and eff_predict > 0:
            options_dict["num_predict"] = int(eff_predict)

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": options_dict,
        }
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive
        if self.think is not None:
            payload["think"] = self.think

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

    def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """
        Send streaming chat completion request to local Ollama.
        Yields cleaned token chunks in real-time, statefully suppressing <think> blocks.

        Args:
            system_prompt: Grounded system instruction.
            user_prompt: User question with retrieved context.
            temperature: Optional override for temperature.
            num_predict: Optional override for max tokens.

        Yields:
            Cleaned token text pieces.
        """
        url = f"{self.base_url}/api/chat"
        temp = self.temperature if temperature is None else float(temperature)
        eff_predict = num_predict if num_predict is not None else self.num_predict

        options_dict: dict[str, Any] = {"temperature": temp}
        if eff_predict is not None and eff_predict > 0:
            options_dict["num_predict"] = int(eff_predict)

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": True,
            "options": options_dict,
        }
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive
        if self.think is not None:
            payload["think"] = self.think

        client = self._get_client()
        filter_think = StreamThinkFilter()
        start_time = time.perf_counter()
        first_token_recorded = False
        self.last_ttft_ms = None

        try:
            with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    raw_content = data.get("message", {}).get("content", "")
                    clean_piece = filter_think.feed(raw_content)

                    if clean_piece:
                        if not first_token_recorded:
                            self.last_ttft_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
                            first_token_recorded = True
                        yield clean_piece

                    if data.get("done"):
                        metric_keys = [
                            "total_duration",
                            "load_duration",
                            "prompt_eval_count",
                            "prompt_eval_duration",
                            "eval_count",
                            "eval_duration",
                        ]
                        self.last_metrics = {k: data[k] for k in metric_keys if k in data}

                final_piece = filter_think.finalize()
                if final_piece:
                    if not first_token_recorded:
                        self.last_ttft_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
                        first_token_recorded = True
                    yield final_piece

        except httpx.ConnectError as e:
            raise QwenConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. Please ensure Ollama is running."
            ) from e
        except httpx.TimeoutException as e:
            raise QwenTimeoutError(
                f"Streaming request to Ollama at {self.base_url} timed out after {self.timeout}s."
            ) from e
        except httpx.HTTPStatusError as e:
            raise QwenAPIError(
                f"Ollama API returned HTTP error {e.response.status_code}: {e.response.text}"
            ) from e
        except Exception as e:
            raise QwenAPIError(f"Unexpected error streaming from Ollama: {e}") from e
        finally:
            if self._custom_client is None:
                client.close()


def warm_up_model(
    client: Optional[QwenClient] = None,
    timeout: float = 10.0,
) -> bool:
    """
    Optional blocking helper to warm up the local Ollama model on startup.
    Sends a minimal 1-token prompt with short timeout to preload model weights into RAM/VRAM.
    Catches all connection/timeout errors and returns True/False without crashing application startup.
    """
    try:
        target_client = client or QwenClient(timeout=timeout, num_predict=1)
        target_client.generate("hi", "hi", num_predict=1)
        return True
    except Exception:
        return False
