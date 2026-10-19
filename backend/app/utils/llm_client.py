"""
LLM Client Wrapper
Unified OpenAI format API calls
Supports Ollama num_ctx parameter to prevent prompt truncation
"""

import json
import os
import re
from typing import Optional, Dict, Any, List
from openai import APIStatusError, OpenAI
import requests

from ..config import Config


class LLMOutputTruncatedError(ValueError):
    """Raised when the provider stops generation at the output-token limit."""

    def __init__(self, message: str, content: str = ""):
        super().__init__(message)
        self.content = content


class LLMClient:
    """LLM Client"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 300.0
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL_NAME
        self.timeout = timeout

        if not self.api_key:
            raise ValueError("LLM_API_KEY not configured")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout,
        )

        # Keep the default context moderate for local hardware; document input is capped separately.
        self._num_ctx = int(os.environ.get('OLLAMA_NUM_CTX', '8192'))

    def _is_ollama(self) -> bool:
        """Check if we're talking to an Ollama server."""
        return '11434' in (self.base_url or '')

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None,
        think: Optional[bool] = None,
    ) -> str:
        """
        Send chat request

        Args:
            messages: Message list
            temperature: Temperature parameter
            max_tokens: Max token count
            response_format: Response format (e.g., JSON mode)

        Returns:
            Model response text
        """
        if self._is_ollama():
            return self._chat_ollama(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                think=think,
            )

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format:
            kwargs["response_format"] = response_format

        # For Ollama: pass num_ctx via extra_body to prevent prompt truncation
        if self._is_ollama() and self._num_ctx:
            kwargs["extra_body"] = {
                "options": {"num_ctx": self._num_ctx}
            }

        try:
            response = self.client.chat.completions.create(**kwargs)
        except APIStatusError as error:
            if not response_format or not self._is_unsupported_response_format(error):
                raise
            kwargs.pop("response_format")
            response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise LLMOutputTruncatedError(
                "LLM response truncated at the output-token limit",
                content=choice.message.content or "",
            )
        content = choice.message.content
        if not isinstance(content, str):
            raise ValueError("LLM response did not contain message content")
        # Some models (like MiniMax M2.5) include <think>thinking content in response, need to remove
        content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
        return content

    @staticmethod
    def _is_unsupported_response_format(error: APIStatusError) -> bool:
        """Recognize provider errors that explicitly reject JSON response mode."""
        details = f"{error} {error.body!r}".lower()
        return (
            "response_format" in details
            and any(
                phrase in details
                for phrase in (
                    "not supported",
                    "unsupported",
                    "unknown parameter",
                    "unrecognized request argument",
                )
            )
        )

    def _chat_ollama(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        response_format: Optional[Dict],
        think: Optional[bool],
    ) -> str:
        """Call Ollama's native API so thinking models can return their final answer."""
        base_url = (self.base_url or '').rstrip('/')
        if base_url.endswith('/v1'):
            base_url = base_url[:-3]

        options: Dict[str, Any] = {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
        if self._num_ctx:
            options["num_ctx"] = self._num_ctx

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        if think is not None:
            payload["think"] = think
        if response_format:
            payload["format"] = (
                "json"
                if response_format.get("type") == "json_object"
                else response_format
            )

        response = requests.post(
            f"{base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        result = response.json()
        if result.get("done_reason") == "length":
            raise ValueError(
                "Ollama stopped at the context or output token limit; "
                "increase OLLAMA_NUM_CTX or reduce the document size"
            )
        content = result.get("message", {}).get("content")
        if not isinstance(content, str):
            raise ValueError("Ollama response did not contain message content")

        return re.sub(r'<think>[\s\S]*?</think>', '', content).strip()

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
        think: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Send chat request and return JSON

        Args:
            messages: Message list
            temperature: Temperature parameter
            max_tokens: Max token count

        Returns:
            Parsed JSON object
        """
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            think=think,
        )
        # Clean markdown code block markers
        cleaned_response = response.strip()
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)
        cleaned_response = cleaned_response.strip()

        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format from LLM: {cleaned_response}")
