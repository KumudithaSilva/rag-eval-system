from abc import ABC, abstractmethod
from typing import List


class IChunkingStrategy(ABC):
    """
    Interface for chunking strategies.
    """

    @abstractmethod
    def chunk(self, documents: list) -> List:
        """
        Split text into chunks.

        Args:
            documents (list): List of documents.

        Returns:
            List: A list containing the chunked documents.
        """
        pass
