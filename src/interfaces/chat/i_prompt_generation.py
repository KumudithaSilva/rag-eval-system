from abc import ABC, abstractmethod
from typing import Dict, List
from langchain_core.documents import Document


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

    @abstractmethod
    def generate_batch(self, documents: List[Document]) -> List[List[Dict]]:
        """
        Generate prompts for a batch of documents.

        Args:
            documents (List[Document]): A list of LangChain source documents to be chunked.

        Returns:
            List[List[Dict]]: A list of chat message batches, where each batch corresponds to a document.
        """
        pass
