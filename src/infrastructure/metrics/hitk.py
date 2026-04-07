from typing import List
from interfaces.metrics.i_retrieval_metric import IRetrievalMetric


class HitAtKMetric(IRetrievalMetric):
    """
    Hit@K metric.

    - Checks whether a keyword appears in the top-K retrieved documents.
    - Returns 1.0 if found, 0.0 otherwise.

    Attributes:
        k (int): Number of top documents to consider.
    """

    def __init__(self, k: int = 10):
        self.k = k

    @property
    def name(self):
        return "hit@K"

    def compute(self, keyword: str, retrieved_docs: List) -> float:
        """
        Check if a single keyword appears in the top-K documents.

        Args:
            keyword (str): Query keyword.
            retrieved_docs (List[str]): List of retrieved document texts.

        Returns:
            float: Keyword found in docs (1.0 if keyword is found, else 0.0).
        """
        keyword_lower = keyword.lower()
        top_docs = retrieved_docs[: self.k]

        for doc in top_docs:
            if keyword_lower in doc.lower():
                return 1.0
        return 0.0

    def compute_for_question(self, keywords: List, retrieved_docs: List) -> float:
        """
        Compute Hit@K for all keywords in a question.

        Args:
            keywords (List[str]): List of keywords for the question.
            retrieved_docs (List[str]): List of retrieved document texts.

        Returns:
            float: Keyword found in docs (1.0 if keyword is found, else 0.0).
        """
        for keyword in keywords:
            if self.compute(keyword, retrieved_docs) == 1.0:
                return 1.0
        return 0.0
