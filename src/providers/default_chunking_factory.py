from infrastructure.chunking.default_chunking import DefaultChunking
from interfaces.chunking.i_chunking_strategy import IChunkingStrategy
from interfaces.infra.i_chunking_factory import IChunkingFactory


class DefaultChunkingFactory(IChunkingFactory):
    """
    Factory responsible for creating DefaultChunking instances.
    """

    def create(self, config: dict, documents: list) -> IChunkingStrategy:
        # Extract configuration parameters
        chunk_size = config.get("chunk_size")
        chunk_overlap = config.get("chunk_overlap")

        # Create strategy
        return DefaultChunking(
            documents=documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
