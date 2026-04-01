from infrastructure.chunking.default_chunking import DefaultChunking
from interfaces.chunking.i_chunking_strategy import IChunkingStrategy
from interfaces.infra.i_chunking_factory import IChunkingFactory


class DefaultChunkingFactory(IChunkingFactory):
    """
    Factory responsible for creating DefaultChunking instances.
    """

    def create(self, config: dict, documents: list) -> IChunkingStrategy:
        # Extract configuration parameters
        k = config.get("k")
        size = config.get("size")

        # Create strategy
        return DefaultChunking(documents=documents, k=k, size=size)
