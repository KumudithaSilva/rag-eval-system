from typing import List
from dto.result import Result
from dto.test_rag_retriever import TestQuestion
from interfaces.infra.i_evaluator import IEvaluator
from interfaces.metrics.i_retrieval_metric import IRetrievalMetric
from logs.logger_singleton import Logger


class RetrievalEvaluator(IEvaluator):
    """
    Evaluate multiple retrieval metrics for a dataset of TestQuestions.

    Attributes:
        metrics: (List[RetrievalMetric]): Metrics that evaluator needs to run.
        dataset_results: (dict): Dictionary for storing multiple metrics results.
    """

    def __init__(
        self,
        metrics: List[IRetrievalMetric],
        logger=None,
    ):
        """
        Initialize the rag evaluator service.

        Args:
            metrics: (List[RetrievalMetric]): Metrics that evaluator needs to run.
            logger (Logger, optional): A logger instance. If None, a default
                logger is created using the class name.
        """
        self.metrics = metrics
        self.dataset_results = {}
        self.logger = logger or Logger(self.__class__.__name__)

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
        results = {}

        for metric in self.metrics:

            total_score = 0.0
            count = 0

            self.logger.info(f"Evaluating metric: {metric.name}")

            for question, retrieved_docs in zip(test_questions, retriver_answers):
                doc_texts = [doc.page_content for doc in retrieved_docs]

                score = metric.compute_for_question(question.keywords, doc_texts)

                total_score += score
                count += 1

                self.logger.info(
                    f"Total score for Question: {count} | Metric: '{metric.name}': {score:.4f}\n"
                )

                avg_score = total_score / count if count > 0 else 0.0
                results[metric.name] = avg_score

        self.dataset_results = results
        return self.dataset_results
