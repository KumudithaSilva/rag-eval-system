from typing import Optional
from infrastructure.llms.open_router_client import OpenRouterAIClientWrapper
from infrastructure.llms.ollma_client import OllamaClientWrapper
from interfaces.chat.i_chatbot_connection import IChatConnection
from interfaces.infra.i_api_key_provider import IApiKeyProvider
from interfaces.llm.i_ai_client import IAIClient


class ChatConnectionService(IChatConnection):
    """
    Initializes AI chatbot connection using interface-based dependencies.
    """

    def __init__(self, key_provider: IApiKeyProvider):
        """
        Initialize ChatConnectionService.

        Args:
            key_provider (IApiKeyProvider): Interface to get API key.
        """
        self.key_provider: IApiKeyProvider = key_provider
        self.client: Optional[IAIClient] = None

    def connect(self, model: str) -> IAIClient:
        """
        Create and return the OpenRouterAI client. Only creates it once per instance.

        Args:
            model (str): The name of the model to connect to.

        Returns:
            IAIClient: Interface for AI client operations.
        """
        if self.client is None:
            self.client = OpenRouterAIClientWrapper(self.key_provider, model=model)
            # self.client = OllamaClientWrapper(model=model)
        return self.client
