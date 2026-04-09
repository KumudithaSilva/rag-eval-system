from abc import ABC, abstractmethod
from typing import List


class IChunkingStrategy(ABC):
    """
    Interface for chunking strategies.
    """

    @abstractmethod
    def chunk(self) -> List:
        """
        Split text into chunks.

        Returns:
            List: A list containing the chunked documents.
        """
        pass
