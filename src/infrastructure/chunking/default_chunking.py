from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from interfaces.chunking.i_chunking_strategy import IChunkingStrategy
from logs.logger_singleton import Logger


class DefaultChunking(IChunkingStrategy):
    """
    Chunk text using a fixed size and overlap.

    Attributes:
        documents (list): Preloaded documents to be chunked.
        chunk_size (int): The size of each chunk.
        chunk_overlap (int): The overlap size between chunks.
        _documents (List | None): Internal cache of loaded documents.
    """

    def __init__(
        self, documents: list, chunk_size: int, chunk_overlap: int, logger=None
    ):
        """
        Initialize the DefaultChunking strategy.

        Args:
            documents (list): Preloaded documents to be chunked.
            chunk_size (int): The size of each chunk.
            chunk_overlap (int): The overlap size between chunks.
            logger (Logger, optional): A logger instance. If None, a default
                logger is created using the class name.
        """
        self.documents = documents
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logger or Logger(self.__class__.__name__)

    def chunk(self) -> List:
        """
        Chunk the given text based on fixed size and overlap.

        Returns:
            List: List of chunked documents.
        """
        if not self.documents:
            return []
        self.logger.debug(
            f"Chunk Size: {self.chunk_size}, Chunk Overlap: {self.chunk_overlap}"
        )
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        chunks = text_splitter.split_documents(self.documents)
        self.logger.debug(f"Generated {chunks[0]} chunks.")
        return chunks
