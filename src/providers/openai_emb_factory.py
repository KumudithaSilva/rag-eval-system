from infrastructure.embedding.openai_embeddings import OpenAIEmbeddingModel
from interfaces.infra.i_api_key_provider import IApiKeyProvider
from interfaces.infra.i_embedding_factory import IEmbeddingFactory


class OpenAIEmbeddingFactory(IEmbeddingFactory):
    """
    Factory responsible for creating OpenAIEmbeddings instances.
    """

    def __init__(self, key_provider: IApiKeyProvider):
        self.key_provider = key_provider

    def create(self, config: dict):
        # Extract configuration parameters
        model_name = config.get("emb_model_name")

        # Initialize key provider to ensure API keys are loaded
        key_provider = self.key_provider

        # Create and return the embedding strategy instance
        return OpenAIEmbeddingModel(model_name=model_name, key_provider=key_provider)
