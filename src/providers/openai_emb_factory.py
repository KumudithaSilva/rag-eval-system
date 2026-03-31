from container.factory_container import FactoryContainer
from infrastructure.embedding.openai_embeddings import OpenAIEmbeddingModel
from interfaces.infra.i_embedding_factory import IEmbeddingFactory
from interfaces.infra.i_factory_container import IFactoryContainer


class OpenAIEmbeddingFactory(IEmbeddingFactory):
    """
    Factory responsible for creating OpenAIEmbeddings instances.
    """

    def __init__(self, factory_container: IFactoryContainer | None = None):
        self.factory_container: IFactoryContainer = (
            factory_container or FactoryContainer()
        )

    def create(self, config: dict):
        # Extract configuration parameters
        model_name = config.get("emb_model_name")

        # Validate required parameters
        if not model_name:
            raise ValueError("Missing required parameter: model_name")

        # Initialize key provider to ensure API keys are loaded
        openai_key = self.factory_container.create_key_provider()

        # Create and return the embedding strategy instance
        return OpenAIEmbeddingModel(model_name=model_name, key_provider=openai_key)
