from abc import ABC, abstractmethod
from typing import List

from dto.result import Result
from dto.test_rag_retriever import TestQuestion


class IEvaluator(ABC):
    """
    Interface for initializing RAG Testset Evaluator.
    """

    @abstractmethod
    def evaluate(
        self, test_questions: List[TestQuestion], retriver_answers: List[List[Result]]
    ) -> List[dict]:
        """
        Evaluate Metrics for all questions in the test set.

        Args:
            test_questions (List[TestQuestion]): List of Test questions
            retriver_answers (List[List[Any]]): Retrieved context documents for questions.

        Return:
            List[dict]: Evaluated results for multiple metrics.
        """
        pass
