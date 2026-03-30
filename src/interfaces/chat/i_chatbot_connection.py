from abc import ABC, abstractmethod
from interfaces.llm.i_ai_client import IAIClient


class IChatConnection(ABC):
    """
    Interface for initializing AI chatbot connection.
    """

    @abstractmethod
    def connect(self, model: str) -> IAIClient:
        """
        Initializing AI chatbot converstion.

        Args:
            model (str): The name of the model to connect to.

        Returns:
            IAIClient: Interface for AI client operations.
        """
        pass
