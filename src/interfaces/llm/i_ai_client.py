from abc import ABC, abstractmethod
from typing import Dict, List


class IAIClient(ABC):
    """
    Abstract interface for an AI client.
    """

    @abstractmethod
    def chat_completions_create(
        self, messages: List[Dict], structured: bool = True
    ) -> str:
        """
        Sends a chat completion request to the AI backend.

        Args:
            messages: List of chat messages
            structured: If True, returns structured output; otherwise plain text.

        Returns:
            The AI-generated response text.
        """
        pass
