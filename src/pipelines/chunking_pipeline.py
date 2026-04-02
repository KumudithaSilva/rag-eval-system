from interfaces.chunking.i_chunking_strategy import IChunkingStrategy
from interfaces.infra.i_pipeline import IPipelineStep
from logs.logger_singleton import Logger


class ChunkingStep(IPipelineStep):
    """
    Pipeline step for chunking documents.

    args:
        chunking_strategy (IChunkingStrategy): The strategy to use for chunking documents.
    """

    def __init__(self, chunking_strategy: IChunkingStrategy, logger=None):
        """
        Initialize ChunkingStep with a chunking strategy.

        Args:
            chunking_strategy (IChunkingStrategy): The strategy to use for chunking documents.
            logger (Logger, optional): A logger instance. If None, a default
                logger is created using the class name.
        """
        self.chunking = chunking_strategy
        self.logger = logger or Logger(self.__class__.__name__)

    def run(self, data: dict):
        """
        Run the chunking step on the input data.

        args:
            data (dict): Store the results of the chunking process.

        returns:
            dict: The input data with the added chunks.
        """
        chunks = self.chunking.chunk()
        data["chunks"] = chunks

        self.logger.info("ChunkingStep completed successfully")
        self.logger.debug(f"Data Store {data}")

        return data
