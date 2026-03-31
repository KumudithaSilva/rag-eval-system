from interfaces.embedding.i_embedding import IEmbeddingModel
from registry.embedding_registry import EMBEDDING_REGISTRY


class EmbeddingFactory:
    """Create embedding model instances."""

    @staticmethod
    def create(embedding_data: dict) -> IEmbeddingModel:
        embedding_type = embedding_data.get("type")
        config = embedding_data.get("config", {})

        factory = EMBEDDING_REGISTRY.get(embedding_type)

        if not factory:
            raise ValueError(f"Unsupported embedding type: {embedding_type}")

        return factory.create(config)
