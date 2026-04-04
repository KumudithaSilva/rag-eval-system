import math

from interfaces.metrics.i_retrieval_metric import RetrievalMetric


class NDCGMetric(RetrievalMetric):
    """Normalized Discounted Cumulative Gain.

    Attributes:
        k (int): Rank cutoff.
    """

    def __init__(self, k: int = 10):
        self.k = k

    def compute(self, keyword: str, retrieved_docs: list) -> float:
        """Compute nDCG score.

        Args:
            keyword (str): Query keyword.
            retrieved_docs (list): Retrieved documents.

        Returns:
            float: nDCG score.
        """
        relevances = self._binary_relevance(keyword, retrieved_docs)

        dcg = self._dcg(relevances)
        idcg = self._dcg(sorted(relevances, reverse=True))

        return dcg / idcg if idcg > 0 else 0.0

    def _binary_relevance(self, keyword: str, docs: list) -> list[int]:
        """Generate binary relevance scores."""
        keyword_lower = keyword.lower()

        return [
            1 if keyword_lower in doc.page_content.lower() else 0
            for doc in docs[: self.k]
        ]

    def _dcg(self, relevances: list[int]) -> float:
        """Compute Discounted Cumulative Gain."""
        return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[: self.k]))
