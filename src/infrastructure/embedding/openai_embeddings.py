from langchain_openai import OpenAIEmbeddings
from interfaces.embedding.i_embedding import IEmbeddingModel
from interfaces.infra.i_api_key_provider import IApiKeyProvider
from logs.logger_singleton import Logger


class OpenAIEmbeddingModel(IEmbeddingModel):
    """
    Concrete OpenAIEmbedding.

    Attributes:
        key_provider (IApiKeyProvider): An instance of IApiKeyProvider to retrieve the API key.
        _embedding_model (OpenAIEmbeddingModel): The OpenAI embedding model instance.
    """

    def __init__(self, model_name: str, key_provider: IApiKeyProvider, logger=None):
        """
        Initialize the OpenAI embedding model.

        Args:
            model_name: The name of the OpenAI model to use for embeddings.
            key_provider: An instance of IApiKeyProvider to retrieve the API key.
            logger (Logger, optional): A logger instance. If None, a default
                logger is created using the class name.
        """
        self.key_provider = key_provider.get_api_key()
        self._embedding_model = OpenAIEmbeddings(
            model=model_name, api_key=self.key_provider
        )
        self.logger = logger or Logger(self.__class__.__name__)

    def get_model(self):
        """
        Get the underlying OpenAI embedding model instance.

        Return:
            The OpenAI embedding model instance.
        """
        try:
            test_text = "Hello world"
            vector = self._embedding_model.embed_query(test_text)
            self.logger.debug(f"Returning OpenAI Embedding vectors: {vector[:5]}")
        except Exception as e:
            self.logger.warning(f"Could not retrieve Embedding vectors: {e}")

        return self._embedding_model
