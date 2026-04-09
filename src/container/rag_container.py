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
from pipelines.chunking_pipeline import ChunkingStep
from pipelines.mongo_storage_pipeline import MongoStoreStep
from pipelines.rag_test_evaluator_pipeline import EvaluatorStep
from pipelines.rag_test_retriever_pipeline import RetriverStep
from pipelines.vector_store_pipeline import VectorStoreStep


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
        self.embedding_factory = EmbeddingFactory(
            self.app_container.get_embedding_factory_registry()
        )

    def create_mongo_connection_service(self):
        """
        Create and return instance of MongoConnection.

        Returns:
            MongoConnection: Mongo connection service instance.
        """
        mongo_connection = self.app_container.create_mongo_connection_service()
        return mongo_connection

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

    def create_pipeline(self, document_config: dict, path: str, testset_path: str):
        """
        Create and return a RAG pipeline instance.

        Args:
            document_data (dict): Document data containing chunking configuration.
            path (str): Path to the documents.
            testset_path (str): Path to the rag test documents.

        Returns:
            RAGPipeline: RAG pipeline instance.
        """
        documents = self.app_container.doc_service(path)[:10]
        rag_tests = self.app_container.rag_test_service(testset_path)[:10]
        config_details = document_config

        chunking = self.chunking_factory.create(document_config["chunking"], documents)
        embedding_model = self.embedding_factory.create(document_config["embedding"])

        vector_store_service = self.app_container.get_vector_store_service(
            embedding_model
        )

        rag_test_retriver = self.app_container.get_rag_test_retriever(
            vector_store_service
        )

        rag_test_evaluator = self.app_container.get_rag_evaluator()

        mongo_connection = self.app_container.create_mongo_connection_service()

        steps = []

        steps.append(ChunkingStep(chunking))
        steps.append(VectorStoreStep(vector_store_service))
        steps.append(RetriverStep(rag_test_retriver, rag_tests))
        steps.append(EvaluatorStep(rag_test_evaluator, rag_tests))
        steps.append(MongoStoreStep(mongo_connection, config_details))

        return RAGPipeline(steps)
