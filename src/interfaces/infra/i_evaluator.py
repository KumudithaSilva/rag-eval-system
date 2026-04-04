from abc import ABC, abstractmethod
from typing import List


class IEvaluator(ABC):
    """
    Interface for initializing RAG Testset Evaluator.
    """

    @abstractmethod
    def evaluate() -> List[dict]:
        """
        Evaluate Metrics for all questions in the test set.

        Return:
            List[dict]: Evaluated results for multiple metrics.
        """
        pass
