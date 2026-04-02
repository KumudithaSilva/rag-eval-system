from interfaces.chunking.i_chunking_strategy import IChunkingStrategy
from interfaces.infra.i_pipeline import IPipelineStep


class ChunkingStep(IPipelineStep):
    """
    Pipeline step for chunking documents.

    args:
        chunking_strategy (IChunkingStrategy): The strategy to use for chunking documents.
    """

    def __init__(self, chunking_strategy: IChunkingStrategy):
        """
        Initialize ChunkingStep with a chunking strategy.

        Args:
            chunking_strategy (IChunkingStrategy): The strategy to use for chunking documents.
        """
        self.chunking = chunking_strategy

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
        return data
