from langchain_huggingface import HuggingFaceEmbeddings
from infrastructure.logs.logger_singleton import Logger
from interfaces.embedding.i_embedding import IEmbeddingModel


class HuggingFaceEmbeddingModel(IEmbeddingModel):
    """
    Concrete HuggingFaceEmbeddings.

    Attributes:
        _embedding_model (HuggingFaceEmbeddings): The HuggingFace embedding model instance.
    """

    def __init__(self, model_name: str, logger=None):
        """
        Initialize the HuggingFace embedding model.

        Args:
            model_name: The name of the HuggingFace model to use for embeddings.
            logger (Logger, optional): A logger instance. If None, a default
                logger is created using the class name.
        """
        self._embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        self.logger = logger or Logger(self.__class__.__name__)

    def get_model(self):
        """
        Get the underlying HuggingFace embedding model instance.

        Return:
            The HuggingFace embedding model instance.
        """
        try:
            test_text = "Hello world"
            vector = self._embedding_model.embed_query(test_text)
            self.logger.debug(f"Returning HuggingFace Embedding vectors: {vector[:5]}")
        except Exception as e:
            self.logger.warning(f"Could not retrieve Embedding vectors: {e}")

        return self._embedding_model
