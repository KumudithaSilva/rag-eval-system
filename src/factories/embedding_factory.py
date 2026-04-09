from interfaces.embedding.i_embedding import IEmbeddingModel
from registry.embedding_registry import EmbeddingFactoryRegistry


class EmbeddingFactory:
    """Create dynamic embedding model instances."""

    def __init__(self, registry: EmbeddingFactoryRegistry):
        self.registry = registry

    def create(self, embedding_data: dict) -> IEmbeddingModel:
        embedding_type = embedding_data.get("type")
        config = embedding_data.get("config", {})

        factory = self.registry.get(embedding_type)
        return factory.create(config)
