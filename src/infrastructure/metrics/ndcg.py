import math
from typing import List

from interfaces.metrics.i_retrieval_metric import IRetrievalMetric


class NDCGMetric(IRetrievalMetric):
    """
    Normalized Discounted Cumulative Gain.

     - This metric evaluates considering both the relevance of each document and its position in the ranking.
     - Higher relevance documents appearing earlier in the list contribute more to the score.

    Attributes:
        k (int): Rank cutoff.

    Methods:
        compute(keyword, retrieved_docs):
            Computes the nDCG score for a given query keyword and a list of retrieved documents.
        _binary_relevance(keyword, docs):
            Generates a binary relevance list for the top-k documents.
        _dcg(relevances):
            Computes the Discounted Cumulative Gain (DCG) for a list of relevance scores.
    """

    def __init__(self, k: int = 10):
        self.k = k

    @property
    def name(self):
        """nDCG score"""
        return "ndcg"

    def compute(self, keyword: str, retrieved_docs: List) -> float:
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

    def compute_for_question(self, keywords: List, retrieved_docs: List) -> float:
        """Compute average nDCG over all keywords in a TestQuestion.

        Args:
            keywords (TestQuestion): Question with one or more keywords.
            retrieved_docs (list): List of retrieved document texts.

        Returns:
            float: Average nDCG for the question.
        """
        scores = [self.compute(k, retrieved_docs) for k in keywords]
        return sum(scores) / len(scores) if scores else 0.0

    def _binary_relevance(self, keyword: str, docs: list) -> list[int]:
        """Generate binary relevance scores for the top-k documents.

        How it works:
        - Converts the query `keyword` to lowercase for case-insensitive matching.
        - Looks at each document in the top-k retrieved documents.
        - Returns 1 if the keyword is found in the document, 0 otherwise.
        - If there are fewer than k documents, it uses all available ones.

        Example:
            keyword = "apple"
            docs = [doc1, doc2]  # doc1 contains "apple", doc2 does not
            returns [1, 0]
        """
        keyword_lower = keyword.lower()

        return [1 if keyword_lower in doc.lower() else 0 for doc in docs[: self.k]]

    def _dcg(self, relevances: list[int]) -> float:
        """Compute Discounted Cumulative Gain (DCG) for a list of relevance scores.

        How it works:
        - Each relevant document contributes more if it appears earlier in the ranking.
        - The contribution is discounted by the log of its rank: rel / log2(rank + 1)
        - Sums the contributions of the top-k documents.
        - If there are fewer than k relevance scores, it uses all available ones.

        Example:
            relevances = [1, 0, 1]  # first and third docs are relevant
            DCG = 1 / log2(1+1) + 0 / log2(2+1) + 1 / log2(3+1)
                ≈ 1 + 0 + 0.5 = 1.5
        """
        return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[: self.k]))
