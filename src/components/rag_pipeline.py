from interfaces.chunking.i_chunking_strategy import IChunkingStrategy
from interfaces.embedding.i_embedding import IEmbeddingModel
from interfaces.infra.i_pipeline import IPipelineStep
from interfaces.infra.i_rag_pipeline import IRagPipeline
from interfaces.infra.i_vector_store import IVectorStoreService
from logs.logger_singleton import Logger


class RAGPipeline(IRagPipeline):
    """Pipeline for processing workflow."""

    def __init__(self, steps: list[IPipelineStep], logger=None):
        """
        Initialize RAGPipeline with procedure workflow.

        Args:
            steps (list[IPipelineStep]): The list of procedures.
            logger (Logger, optional): Logger instance. If not provided,
                a default logger is created using the class name.
        """
        self.steps = steps
        self.logger = logger or Logger(self.__class__.__name__)

    def process(self, input_data: dict = None):
        """
        Process the rag pipeline.

        Args:
            input_data (dict) : Input data for the pipeline step.
        """
        data = input_data or {}

        for step in self.steps:
            data = step.run(data)

        self.logger.debug(f"Generated Data : {data}")
        return data
