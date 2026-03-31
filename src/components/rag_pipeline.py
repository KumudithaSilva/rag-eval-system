from interfaces.chunking.i_chunking_strategy import IChunkingStrategy
from interfaces.embedding.i_embedding import IEmbeddingModel
from interfaces.infra.i_rag_pipeline import IRagPipeline


class RAGPipeline(IRagPipeline):
    """Pipeline for processing documents."""

    def __init__(
        self, chunking_strategy: IChunkingStrategy, embedding_model: IEmbeddingModel
    ):
        """
        Initialize RAGPipeline with a chunking strategy and embeddings.

        Args:
            chunking_strategy (IChunkingStrategy): The strategy to use for chunking documents.
            embedding_model (IEmbeddingModel): The embedding model to use for vectorization.
        """
        self.chunking = chunking_strategy
        self.embedding = embedding_model

    def process(self):
        """Process the document using the chunking strategy."""
        chunks = self.chunking.chunk()
        print(f"Chunks: {chunks}")

        embeddings = self.embedding.get_model()
        print(f"Embeddings: {embeddings}")
        return {"chunks": chunks}
