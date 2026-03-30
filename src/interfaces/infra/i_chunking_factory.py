from abc import ABC, abstractmethod
from interfaces.chunking.i_chunking_strategy import IChunkingStrategy


class IChunkingFactory(ABC):
    """
    Interface for chunking factory.

    Args:
        config (dict): A dictionary containing the configuration for the chunking strategy.
        path (str): The path to the data that needs to be chunked.

    Returns:
        IChunkingStrategy: An instance of a class that implements the IChunkingStrategy interface
    """

    @abstractmethod
    def create(self, config: dict, path: str) -> IChunkingStrategy:
        pass
