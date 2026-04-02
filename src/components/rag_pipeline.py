from interfaces.chunking.i_chunking_strategy import IChunkingStrategy
from interfaces.embedding.i_embedding import IEmbeddingModel
from interfaces.infra.i_rag_pipeline import IRagPipeline
from interfaces.infra.i_vector_store import IVectorStoreService


class RAGPipeline(IRagPipeline):
    """Pipeline for processing documents."""

    def __init__(
        self,
        chunking_strategy: IChunkingStrategy,
        vector_store_service: IVectorStoreService,
    ):
        """
        Initialize RAGPipeline with a chunking strategy and embeddings.

        Args:
            chunking_strategy (IChunkingStrategy): The strategy to use for chunking documents.
            vector_store_service (IVectorStoreService): The vector store service to use for storing and retrieving vectors.
        """
        self.chunking = chunking_strategy
        self.vector_store_service = vector_store_service

    def process(self):
        """Process the document using the chunking strategy."""
        chunks = self.chunking.chunk()

        vector_store_service = self.vector_store_service.create(chunks=chunks)
        count = vector_store_service._collection.count()

        print(f"Vector Store Stored documents count : {count}")
        return {"count": count}
