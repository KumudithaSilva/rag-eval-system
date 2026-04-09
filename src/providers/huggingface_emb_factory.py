from infrastructure.embedding.hugging_face_embeddings import HuggingFaceEmbeddingModel
from interfaces.infra.i_embedding_factory import IEmbeddingFactory


class HuggingFaceEmbeddingFactory(IEmbeddingFactory):
    """
    Factory responsible for creating HuggingFaceEmbeddings instances.
    """

    def create(self, config: dict):
        # Extract configuration parameters
        model_name = config.get("emb_model_name")

        # Validate required parameters
        if not model_name:
            raise ValueError("Missing required parameter: model_name")

        # Create and return the embedding strategy instance
        return HuggingFaceEmbeddingModel(model_name=model_name)
