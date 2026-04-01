from components.chat_connection import ChatConnectionService
from components.prompt_generation import PromptGenerationService
from infrastructure.infra.chunking_prompt import PromptProvider
from infrastructure.infra.env_loader import DotEnvLoader
from infrastructure.infra.open_router_provider import OpenRouterProvider
from infrastructure.infra.openai_provider import OpenAIApiKeyProvider
from interfaces.infra.i_api_key_provider import IApiKeyProvider
from interfaces.infra.i_chunking_factory import IChunkingFactory
from providers.default_chunking_factory import DefaultChunkingFactory
from providers.llm_chunking_factory import LLMChunkingFactory
from registry.chunking_registry import FactoryRegistry
from utils.document_loader import doc_convert


class ApplicationContainer:
    """
    Application container responsible for wiring services and providing runtime data.

    Arguments:
        - env_loader: Service to load environment variables.
        - prompt_provider: Service to provide prompts for chunking.
        - prompt_service: Service to generate prompts using the provider.
        - factory_registry: Registry to manage chunking factories.
    """

    def __init__(self):
        """
        Initialize the application container with necessary services and registries.
        """
        self.env_loader = DotEnvLoader()
        self.prompt_provider = PromptProvider()
        self.prompt_service = PromptGenerationService(self.prompt_provider)
        self.factory_registry = FactoryRegistry()
        self._register_chunking_factories()

    def _register_chunking_factories(self):
        """
        Register all chunking factories in the factory registry at application startup.
        """
        llm_factory = self.create_llm_chunking_factory()
        default_factory = self.create_default_chunking_factory()

        self.factory_registry.register("LLM Chunking", llm_factory)
        self.factory_registry.register("Default Chunking", default_factory)

    def create_chat_connection_service(self) -> ChatConnectionService:
        """
        Create and return an instance of ChatConnectionService with the appropriate API key provider.

        Returns:
            ChatConnectionService: An instance of the chat connection service.
        """
        key_provider = OpenRouterProvider(self.env_loader)
        return ChatConnectionService(key_provider)

    def openai_key_provider(self) -> IApiKeyProvider:
        """
        Create and return an instance of OpenAIApiKeyProvider.

        Returns:
            IApiKeyProvider: An instance of the OpenAI API key provider.
        """
        return OpenAIApiKeyProvider(self.env_loader)

    def doc_service(self, path: str) -> list:
        """
        Load documents from the specified path and return a list of documents.

        Args:
            path (str): The file path to load documents from.

        Returns:
            list: A list of loaded documents.
        """
        documents = doc_convert(path) if path else []
        return documents[:1]

    def register_chunking_factory(self, chunk_type: str, factory: IChunkingFactory):
        """
        Register a chunking factory in the factory registry.

        Args:
            chunk_type (str): The type of chunking strategy.
            factory (IChunkingFactory): The factory instance to register.

        """
        self.factory_registry.register(chunk_type, factory)

    def create_llm_chunking_factory(self) -> LLMChunkingFactory:
        """
        Create and return an instance of LLMChunkingFactory with the necessary services.

        Returns:
            LLMChunkingFactory: An instance of the LLM chunking factory.
        """

        llm_client = self.create_chat_connection_service()
        return LLMChunkingFactory(
            prompt_service=self.prompt_service, llm_client=llm_client
        )

    def create_default_chunking_factory(self) -> DefaultChunkingFactory:
        """
        Create and return an instance of DefaultChunkingFactory.

        Returns:
            DefaultChunkingFactory: An instance of the default chunking factory.
        """
        return DefaultChunkingFactory()

    def get_factory_registry(self) -> FactoryRegistry:
        """
        Get the factory registry instance.

        Returns:
            FactoryRegistry: The factory registry instance.
        """
        return self.factory_registry
