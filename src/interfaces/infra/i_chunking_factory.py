from abc import ABC, abstractmethod
from interfaces.chunking.i_chunking_strategy import IChunkingStrategy


class IChunkingFactory(ABC):
    """
    Interface for chunking factory.

    Args:
        config (dict): A dictionary containing the configuration for the chunking strategy.
        documents (list): A list of documents to be chunked.

    Returns:
        IChunkingStrategy: An instance of a class that implements the IChunkingStrategy interface
    """

    @abstractmethod
    def create(self, config: dict, documents: list) -> IChunkingStrategy:
        pass
