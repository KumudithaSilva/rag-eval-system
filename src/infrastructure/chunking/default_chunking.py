from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from infrastructure.infra.file_extractor import FileExtractor
from interfaces.chunking.i_chunking_strategy import IChunkingStrategy
from logs.logger_singleton import Logger
from utils.document_loader import doc_convert


class DefaultChunking(IChunkingStrategy):
    """
    Chunk text using a fixed size and overlap.

    Attributes:
        documents (list): Preloaded documents to be chunked.
        k (int): The overlap size between consecutive chunks.
        size (int): The size of each chunk.
        _documents (List | None): Internal cache of loaded documents.
    """

    def __init__(self, documents: list, k: int = 3, size: int = 3, logger=None):
        """
        Initialize the DefaultChunking strategy.

        Args:
            documents (list): Preloaded documents to be chunked.
            k (int): The overlap size between chunks.
            size (int): The size of each chunk.
            logger (Logger, optional): A logger instance. If None, a default
                logger is created using the class name.
        """
        self.documents = documents
        self.k = k
        self.size = size
        self.logger = logger or Logger(self.__class__.__name__)

    def chunk(self) -> List:
        """
        Chunk the given text based on fixed size and overlap.

        Returns:
            List: List of chunked documents.
        """
        if not self.documents:
            return []
        self.logger.debug(f"K: {self.k}, Size: {self.size}")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        chunks = text_splitter.split_documents(self.documents)
        self.logger.debug(f"Generated {chunks[0]} chunks.")
        return chunks
