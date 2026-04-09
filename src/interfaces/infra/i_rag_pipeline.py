from abc import ABC, abstractmethod


class IRagPipeline(ABC):
    """
    Interface for initializing RAG Pipeline.
    """

    @abstractmethod
    def process(self, input_data: dict = None) -> dict:
        """
        Initializing RAG Pipeline.

        Returns:
            dict: Processed output from the RAG pipeline.
        """
        pass
