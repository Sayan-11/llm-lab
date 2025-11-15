from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from openai import OpenAI


class BaseModelClient(ABC):
    """
    Abstract base class for all model backends.
    """

    def __init__(self, name: str, default_params: Optional[Dict[str, Any]] = None):
        self.name = name
        self.default_params = default_params or {}

    @abstractmethod
    def generate(self, prompt: str, **override_params: Any) -> str:
        """
        Generate a completion for the given prompt.
        """
        ...


class OpenAIClient(BaseModelClient):
    """
    Minimal wrapper around OpenAI's Chat Completions API.
    """

    def __init__(
        self,
        name: str,
        default_params: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
    ) -> None:
        super().__init__(name, default_params)
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OPENAI_API_KEY not set in environment and no api_key passed."
            )
        self._client = OpenAI(api_key=key)

    def generate(self, prompt: str, **override_params: Any) -> str:
        """
        Call the model with a simple user prompt and return plain text.
        """
        params: Dict[str, Any] = {**self.default_params, **override_params}

        resp = self._client.chat.completions.create(
            model=self.name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            **params,
        )

        return resp.choices[0].message.content.strip()
