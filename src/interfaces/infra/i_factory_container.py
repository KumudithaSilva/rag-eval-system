from typing import Protocol

from interfaces.chat.i_oneshot_prompt import IPrompt
from interfaces.chat.i_prompt_generation import IPromptGenereateService
from interfaces.infra.i_api_key_provider import IApiKeyProvider
from interfaces.infra.i_env_loader import IEnvLoader


class IFactoryContainer(Protocol):
    """
    Contract for any factory container that wires services.
    """

    def create_llm_connection_service(
        self,
        env_loader: IEnvLoader | None = None,
        key_provider: IApiKeyProvider | None = None,
    ):
        """
        Create and return a service that provides an LLM client.

        Args:
            env_loader (IEnvLoader, optional): Environment loader.
            key_provider (IApiKeyProvider, optional): API key provider.

        Returns:
            ChatConnectionService instance or compatible object.
        """
        pass

    def create_prompt_generation_service(
        self,
        prompt_provider: IPrompt | None = None,
        prompt_generator: IPromptGenereateService | None = None,
    ):
        """
        Create and return a prompt generation service.

        Returns:
            PromptGenerationService instance or compatible object.
        """
        pass

    def create_key_provider(
        self,
        env_loader: IEnvLoader | None = None,
        key_provider: IApiKeyProvider | None = None,
    ):
        """
        Create and return an keyprovider service.

        Args:
            env_loader (IEnvLoader, optional): Environment loader.
            key_provider (IApiKeyProvider, optional): API key provider.

        Returns:
            An instance of a class that implements IApiKeyProvider.
        """
        pass
