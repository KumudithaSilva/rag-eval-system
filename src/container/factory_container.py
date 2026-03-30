from components.chat_connection import ChatConnectionService
from infrastructure.infra.chunking_prompt import PromptProvider
from infrastructure.infra.open_router_provider import OpenRouterProvider
from interfaces.chat.i_oneshot_prompt import IPrompt
from interfaces.chat.i_prompt_generation import IPromptGenereateService
from interfaces.infra.i_api_key_provider import IApiKeyProvider
from interfaces.infra.i_env_loader import IEnvLoader
from infrastructure.infra.env_loader import DotEnvLoader
from components.prompt_generation import PromptGenerationService
from interfaces.infra.i_factory_container import IFactoryContainer


class FactoryContainer(IFactoryContainer):
    """
    Factory to wire all dependencies and return orchestrator service instances.
    """

    def create_llm_connection_service(
        self,
        env_loader: IEnvLoader | None = None,
        key_provider: IApiKeyProvider | None = None,
    ):
        """
        Create and return a LLM connection service.

        Args:
            env_loader (IEnvLoader, optional): Environment loader.
            key_provider (IApiKeyProvider, optional): API key provider.

        Returns:
            ChatConnectionService: An instance of the ChatConnectionService with all dependencies wired.
        """
        if env_loader is None:
            env_loader = DotEnvLoader()

        if key_provider is None:
            key_provider = OpenRouterProvider(env_loader)

        return ChatConnectionService(key_provider)

    def create_prompt_generation_service(
        self,
        prompt_provider: IPrompt | None = None,
        prompt_generator: IPromptGenereateService | None = None,
    ):
        """
        Create and return a Prompt Generation service.

        Args:
            prompt_provider (IPrompt, optional): An instance of a prompt provider.
            prompt_generator (IPromptGenereateService, optional): An instance of a prompt generation service.

        Returns:
            PromptGenerationService: An instance of the PromptGenerationService with all dependencies wired.
        """

        if prompt_provider is None:
            prompt_provider = PromptProvider()

        if prompt_generator is None:
            return PromptGenerationService(prompt_provider)
