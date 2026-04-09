from typing import List

from dto.test_rag_retriever import TestQuestion
from interfaces.infra.i_pipeline import IPipelineStep
from interfaces.infra.i_retriever import IRetriver
from logs.logger_singleton import Logger


class RetriverStep(IPipelineStep):
    """
    Pipeline step for rag test retriever operations.

    args:
        rag_test_retriver (IRetriver): Retriever for fetching relevant context documents.
        rag_tests (List[TestQuestion]): Loaded RAG test questions.
    """

    def __init__(
        self,
        rag_test_retriver: IRetriver,
        rag_tests: List[TestQuestion],
        logger=None,
    ):
        """
        Initialize ChunkingStep with a chunking strategy.

        Args:
            rag_test_retriver (IRetriver): Retriever for fetching relevant context documents.
            rag_tests (List[TestQuestion]): Loaded RAG test questions.
            logger (Logger, optional): A logger instance. If None, a default
                logger is created using the class name.
        """
        self.rag_test_retriver = rag_test_retriver
        self.rag_tests = rag_tests
        self.logger = logger or Logger(self.__class__.__name__)

    def run(self, data: dict):
        """
        Run the rag test retriever step on the input data.

        args:
            data (dict): Store the results of the rag test retriever.
        """
        retriver = self.rag_test_retriver.fetch_contexts(self.rag_tests)
        data["retriver"] = retriver

        self.logger.info("RetriverStepStep completed successfully")
        self.logger.debug(f"Data Store {data}")

        return data
