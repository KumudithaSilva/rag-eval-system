from components.knowledge_base_generation import KnowledgeBaseGenerationService
from components.mongo_connection import MongoConnectionService
from components.rag_pipeline import RAGPipeline
from container.application_container import ApplicationContainer
from factories.chunking_factory import ChunkingFactory
from factories.embedding_factory import EmbeddingFactory
from infrastructure.infra.file_extractor import FileExtractor
from interfaces.infra.i_api_key_provider import IApiKeyProvider
from interfaces.infra.i_env_loader import IEnvLoader
from infrastructure.infra.mongo_url_provider import MongoUrlProvider
from infrastructure.infra.env_loader import DotEnvLoader
from interfaces.infra.i_file_extractor import IFileExtractor


class RagEvalContainer:
    """
    Factory to wire all dependencies and return orchestrator service instances.
    """

    def __init__(self, app_container: ApplicationContainer):
        """
        Initialize the RAG evaluation container with the application container."
        """
        self.app_container = app_container
        self.chunking_factory = ChunkingFactory(
            self.app_container.get_factory_registry()
        )

    def create_mongo_connection_service(
        self,
        env_loader: IEnvLoader | None = None,
        key_provider: IApiKeyProvider | None = None,
    ):
        """
        Create and return a Mongo connection service.

        Args:
            env_loader (IEnvLoader, optional): Environment loader.
            key_provider (IApiKeyProvider, optional): Mongo URL provider.

        Returns:
            IMongoConnection: Mongo connection service instance.
        """
        if env_loader is None:
            env_loader = DotEnvLoader()

        if key_provider is None:
            key_provider = MongoUrlProvider(env_loader)

        return MongoConnectionService(key_provider)

    def knowledge_base_service(self, file_extractor: IFileExtractor | None = None):
        """
        Create and return a knowledge base generation service.

        Args:
            file_extractor (IFileExtractor, optional): File extractor instance.

        Returns:
            KnowledgeBaseGenerationService: Knowledge base generation service instance.
        """
        if file_extractor is None:
            file_extractor = FileExtractor()

        return KnowledgeBaseGenerationService(file_extractor)

    def create_pipeline(self, document_config: dict, path: str):
        """
        Create and return a RAG pipeline instance.

        Args:
            document_data (dict): Document data containing chunking configuration.
            path (str): Path to the documents.

        Returns:
            RAGPipeline: RAG pipeline instance.
        """
        documents = self.app_container.doc_service(path)

        chunking = self.chunking_factory.create(document_config["chunking"], documents)
        embeddings = EmbeddingFactory.create(document_config["embedding"])

        return RAGPipeline(chunking, embeddings)
