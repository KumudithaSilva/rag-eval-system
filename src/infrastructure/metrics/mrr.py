from typing import List
from interfaces.metrics.i_retrieval_metric import IRetrievalMetric


class MRRMetric(IRetrievalMetric):
    """Mean Reciprocal Rank for a single query."""

    @property
    def name(self):
        """MRR score"""
        return "mrr"

    def compute(self, keyword: str, retrieved_docs: List) -> float:
        """Compute MRR score.

        Args:
            keyword (str): Query keyword.
            retrieved_docs (list): Retrieved documents.

        Returns:
            float: Reciprocal rank score.
        """
        keyword_lower = keyword.lower()

        for rank, doc in enumerate(retrieved_docs, start=1):
            if keyword_lower in doc.lower():
                return 1.0 / rank

        return 0.0

    def compute_for_question(self, keywords: List, retrieved_docs: List) -> float:
        """Compute average MRR over all keywords in a TestQuestion.

        Args:
            keywords (TestQuestion): Question with one or more keywords.
            retrieved_docs (list): List of retrieved document texts.

        Returns:
            float: Average MRR for the question.
        """
        scores = [self.compute(k, retrieved_docs) for k in keywords]
        return sum(scores) / len(scores) if scores else 0.0
