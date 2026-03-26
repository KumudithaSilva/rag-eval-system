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
        path (str): Path to knowledge base.
        k (int): The overlap size between consecutive chunks.
        size (int): The size of each chunk.
        _documents (List | None): Internal cache of loaded documents.
    """

    def __init__(self, path: str, k: int = 3, size: int = 3, logger=None):
        """
        Initialize the DefaultChunking strategy.

        Args:
            path (str): Dynamic document path
            k (int): The overlap size between chunks.
            size (int): The size of each chunk.
            logger (Logger, optional): A logger instance. If None, a default
                logger is created using the class name.
        """
        self.path = path
        self.k = k
        self.size = size
        self._documents: List | None = None
        self.logger = logger or Logger(self.__class__.__name__)

    @property
    def documents(self) -> List:
        """
        Lazy-load and cache documents from the specified path.

        Returns:
            List: A list of document objects loaded from the knowledge base.
        """
        if self._documents is None:
            self._documents = doc_convert(self.path)
        return self._documents

    def chunk(self, documents: List | None = None) -> List:
        """
        Chunk the given text based on fixed size and overlap.

        Args:
            documents (List | None): A list of document objects to be chunked.

        Returns:
            List: List of chunked documents.
        """
        documents = documents or self.documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        return chunks


if __name__ == "__main__":
    extractor = FileExtractor()

    test_archive_path = "knowledge-base.zip"

    try:
        extracted = extractor.extract_file(test_archive_path)
        print(f"Extracted {len(extracted)} files into memory.")

        output_folder = extractor.extract_file_to_folder()
        print(f"Files saved to: {output_folder}")

        default_chunking = DefaultChunking(path=output_folder)
        print(len(default_chunking.chunk()))

    except Exception as e:
        print(f"Error: {e}")
