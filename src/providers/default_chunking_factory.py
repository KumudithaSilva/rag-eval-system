from infrastructure.chunking.default_chunking import DefaultChunking
from interfaces.chunking.i_chunking_strategy import IChunkingStrategy
from interfaces.infra.i_chunking_factory import IChunkingFactory
from utils.document_loader import doc_convert


class DefaultChunkingFactory(IChunkingFactory):
    """
    Factory responsible for creating DefaultChunking instances.
    """

    def create(self, config: dict, path: str) -> IChunkingStrategy:
        # Add path for document loading
        path = path
        # Extract configuration parameters
        k = config.get("k")
        size = config.get("size")

        # Validate required parameters

        # Load documents
        documents = doc_convert(path) if path else []

        # Create strategy
        return DefaultChunking(documents=documents, k=k, size=size)
