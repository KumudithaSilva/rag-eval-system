from abc import ABC, abstractmethod
from typing import Any, Dict, List

from dto.test_rag_retriever import TestQuestion


class IRetriver(ABC):
    """
    Interface for initializing RAG Testset retriver.
    """

    @abstractmethod
    def fetch_context(question: str) -> List:
        """
        Retrieve relevant context documents for a question.

        Args:
            question (str): Test question

        Return:
            List: Retrieved relevant context documents.
        """
        pass

    @abstractmethod
    def fetch_contexts(questions: List[TestQuestion]) -> List[List[Any]]:
        """
        Retrieve relevant context documents for a questions.

        Args:
            questions (List[TestQuestion]): List of Test questions

        Return:
            List[List[Any]]: Retrieved context documents per question.
            Ex: [[doc1, doc2], [doc3, doc4]]

        """
        pass
