from abc import ABC, abstractmethod
from typing import List


class IPromptGenereateService(ABC):
    """
    Interface prompt generation.
    """

    @abstractmethod
    def generate(self, document: List) -> List:
        """
        Generate chat messages combining system and user prompts.

        Returns:
            List: Return combined system and user prompt with documents.
        """
        pass
