from abc import ABC, abstractmethod

from core.llm.types import LLMRequest, LLMResponse, ProviderHealth


class LLMProvider(ABC):
    provider_name: str
    model: str

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError

    @abstractmethod
    async def health(self) -> ProviderHealth:
        raise NotImplementedError
