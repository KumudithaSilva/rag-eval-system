from typing import List

from dto.test_rag_retriever import TestQuestion
from interfaces.infra.i_evaluator import IEvaluator
from interfaces.infra.i_pipeline import IPipelineStep
from interfaces.infra.i_retriever import IRetriver
from logs.logger_singleton import Logger


class EvaluatorStep(IPipelineStep):
    """
    Pipeline step for rag test evaluation operations.

    args:
        retrieval_evaluator (IEvaluator): Evaluator for RAG test questions.
        rag_tests (List[TestQuestion]): Loaded RAG test questions.
    """

    def __init__(
        self,
        retrieval_evaluator: IEvaluator,
        rag_tests: List[TestQuestion],
        logger=None,
    ):
        """
        Initialize ChunkingStep with a chunking strategy.

        Args:
            retrieval_evaluator (IEvaluator): Evaluator for RAG test questions.
            rag_tests (List[TestQuestion]): Loaded RAG test questions.
            logger (Logger, optional): A logger instance. If None, a default
                logger is created using the class name.
        """
        self.retrieval_evaluator = retrieval_evaluator
        self.rag_tests = rag_tests
        self.logger = logger or Logger(self.__class__.__name__)

    def run(self, data: dict):
        """
        Run the rag test evaluator step on the input data.

        args:
            data (dict): Store the results of the rag test retriever.
        """
        evaluator = self.retrieval_evaluator.evaluate(
            test_questions=self.rag_tests, retriver_answers=data["retriver"]
        )

        data["evaluator"] = evaluator

        self.logger.info("EvaluatorStep completed successfully")
        self.logger.debug(f"Data Store {data}")

        return data
