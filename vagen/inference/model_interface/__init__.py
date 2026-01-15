from __future__ import annotations

from typing import Any, Dict

REGISTERED_MODEL: Dict[str, Dict[str, Any]] = {}

# Optional providers: each one may have extra dependencies (vllm/openai/anthropic/etc).
# We register only the providers that can be imported in the current environment.

try:
from .vllm import VLLMModelInterface, VLLMModelConfig

    REGISTERED_MODEL["vllm"] = {
        "model_cls": VLLMModelInterface,
        "config_cls": VLLMModelConfig,
    }
except ImportError:
    pass

try:
    from .openai import OpenAIModelInterface, OpenAIModelConfig

    REGISTERED_MODEL["openai"] = {
        "model_cls": OpenAIModelInterface,
        "config_cls": OpenAIModelConfig,
    }
except ImportError:
    pass

try:
    from .claude import ClaudeModelInterface, ClaudeModelConfig

    REGISTERED_MODEL["claude"] = {
        "model_cls": ClaudeModelInterface,
        "config_cls": ClaudeModelConfig,
    }
except ImportError:
    pass

try:
    from .gemini import GeminiModelInterface, GeminiModelConfig

    REGISTERED_MODEL["gemini"] = {
        "model_cls": GeminiModelInterface,
        "config_cls": GeminiModelConfig,
    }
except ImportError:
    pass

try:
    from .routerapi import RouterAPIModelInterface, RouterAPIModelConfig

    REGISTERED_MODEL["routerapi"] = {
        "model_cls": RouterAPIModelInterface,
        "config_cls": RouterAPIModelConfig,
    }
except ImportError:
    pass

try:
    from .together import TogetherModelInterface, TogetherModelConfig

    REGISTERED_MODEL["together"] = {
        "model_cls": TogetherModelInterface,
        "config_cls": TogetherModelConfig,
    }
except ImportError:
    pass