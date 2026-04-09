from abc import ABC, abstractmethod


class IEmbeddingModel(ABC):
    """
    Interface for embedding models.
    """

    @abstractmethod
    def get_model(self):
        """
        Return the underlying embedding model instance.

        Returns:
            The embedding model instance.
        """
        pass
