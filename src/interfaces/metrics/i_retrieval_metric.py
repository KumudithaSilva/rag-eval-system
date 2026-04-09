from abc import ABC, abstractmethod
from typing import List


class IRetrievalMetric(ABC):
    """Base interface for retrieval metrics."""

    @abstractmethod
    def compute(self, keyword: str, retrieved_docs: List) -> float:
        """Compute metric score.

        Args:
            keyword (str): RAG Test Query keyword.
            retrieved_docs (list): List of retrieved documents.

        Returns:
            float: Metric score.
        """
        pass

    @abstractmethod
    def compute_for_question(self, keywords: List, retrieved_docs: List) -> float:
        """Compute average MRR over all keywords in a TestQuestion.

        Args:
            keywords (TestQuestion): Question with one or more keywords.
            retrieved_docs (list): List of retrieved document texts.

        Returns:
            float: Average MRR for the question.
        """
        pass
