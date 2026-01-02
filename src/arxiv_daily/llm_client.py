from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field, ValidationError
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class LLMConfig(BaseModel):
    """Configuration for Large Language Model (LLM) usage."""
    model: str = Field(..., description="Name of the model to use.")
    model_provider: str = Field(..., description="Provider of the model.")
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Model temperature for controlling randomness."
    )
    max_tokens: int = Field(
        default=4096,
        description="Maximum number of output tokens."
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts for failed requests."
    )
    reasoning: Optional[bool] = Field(
        default=None,
        description="Controls the reasoning/thinking mode for supported models."
    )
    base_url: Optional[str] = Field(
        default=None,
        description="Custom API endpoint URL (optional). If not provided, uses provider's default."
    )

_GLOBAL_LLM_CONFIG: Optional[LLMConfig] = None

def basicConfig(**kwargs) -> None:
    """
    Initialize global LLM configuration.

    Args:
        **kwargs: Keyword arguments matching LLMConfig fields.
    """
    global _GLOBAL_LLM_CONFIG
    try:
        _GLOBAL_LLM_CONFIG = LLMConfig(**kwargs)
        logger.debug("Initializing LLM with config: %s", _GLOBAL_LLM_CONFIG.model_dump())
    except ValidationError as e:
        raise ValueError(f"Invalid LLM configuration: {e}") from e


def getLLM() -> BaseChatModel:
    """
    Must be called after basicConfig(). Uses LangChain's init_chat_model() with parameters from the stored LLMConfig.
    """
    if _GLOBAL_LLM_CONFIG is None:
        raise RuntimeError(
            "LLM configuration not initialized. "
            "Call basicConfig() before getLLM()."
        )

    return init_chat_model(**_GLOBAL_LLM_CONFIG.model_dump())
