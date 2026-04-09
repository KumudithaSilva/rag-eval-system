from abc import ABC, abstractmethod
from interfaces.embedding.i_embedding import IEmbeddingModel


class IEmbeddingFactory(ABC):
    """
    Interface for embedding factory.

    Args:
        config (dict): A dictionary containing the configuration for the embedding strategy.

    Returns:
        IEmbeddingModel: An instance of a class that implements the IEmbeddingModel interface
    """

    @abstractmethod
    def create(self, config: dict) -> IEmbeddingModel:
        pass
