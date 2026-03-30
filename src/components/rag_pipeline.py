from interfaces.chunking.i_chunking_strategy import IChunkingStrategy
from interfaces.infra.i_rag_pipeline import IRagPipeline


class RAGPipeline(IRagPipeline):
    """Pipeline for processing documents."""

    def __init__(self, chunking_strategy: IChunkingStrategy):
        """
        Initialize RAGPipeline with a chunking strategy.

        Args:
            chunking_strategy (IChunkingStrategy): The strategy to use for chunking documents.
        """
        self.chunking = chunking_strategy

    def process(self):
        """Process the document using the chunking strategy."""
        chunks = self.chunking.chunk()
        print(f"Chunks: {chunks}")
        return {"chunks": chunks}
