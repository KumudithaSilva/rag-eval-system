from typing import List
from interfaces.metrics.i_retrieval_metric import IRetrievalMetric


class RecallMetric(IRetrievalMetric):
    """
    Keyword-based Recall@K.

    - Measures the fraction of query keywords found in the top-K retrieved documents.
    - Useful for evaluating coverage of relevant information.

    Attributes:
        k (int): Number of top documents to consider.
    """

    def __init__(self, k: int = 10):
        self.k = k

    @property
    def name(self):
        return "recall@K"

    def compute(self, keyword: str, retrieved_docs: List) -> float:
        """
        Check if a keyword is present in the top-K documents.

        Args:
            keyword (str): Query keyword.
            retrieved_docs (list): Retrieved documents.

        Returns:
            float: Recall value for a single keyword.
        """
        keyword_lower = keyword.lower()

        for doc in retrieved_docs:
            if keyword_lower in doc.lower():
                return 1.0
        return 0.0

    def compute_for_question(self, keywords: List, retrieved_docs: List) -> float:
        """
        Compute average recall over all keywords.

        Args:
            keywords (TestQuestion): Question with one or more keywords.
            retrieved_docs (list): List of retrieved document texts.

        Returns:
            float: Recall rank score for the question.
        """
        socres = [self.compute(k, retrieved_docs) for k in keywords]
        recall = sum(socres) / len(socres)

        return recall
