from abc import ABC, abstractmethod


class IApiKeyProvider(ABC):

    @abstractmethod
    def get_api_key(self) -> str:
        pass
