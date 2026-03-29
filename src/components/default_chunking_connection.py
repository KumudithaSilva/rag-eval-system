from typing import List
from infrastructure.chunking.default_chunking import DefaultChunking
from interfaces.chunking.i_chunking_strategy import IChunkingStrategy
from utils.document_loader import doc_convert


class DefaultChunkingConnection(IChunkingStrategy):
    """
    Initialize the DefaultChunking strategy and load documents.

    Args:
        **config: Configuration dictionary containing:
            - path (str): Path to the documents.
            - k (int, optional): Chunk overlap size.
            - size (int, optional): Chunk size.
            - logger (optional): Logger instance.
    """

    def __init__(self, **config):
        # Extract path separately
        path = config.pop("path", None)

        # Load documents from path
        documents = doc_convert(path) if path else []

        # Initialize DefaultChunking with only valid arguments
        self.instance = DefaultChunking(documents=documents, **config)

    def chunk(self) -> List:
        """
        Chunk documents using DefaultChunking strategy.

        Returns:
            List: List of chunked documents.
        """
        return self.instance.chunk()


if __name__ == "__main__":
    from infrastructure.infra.file_extractor import FileExtractor

    extractor = FileExtractor()

    test_archive_path = "knowledge-base.zip"

    try:
        extracted = extractor.extract_file(test_archive_path)
        print(f"Extracted {len(extracted)} files into memory.")

        output_folder = extractor.extract_file_to_folder()
        print(f"Files saved to: {output_folder}")

        config = {"k": 2, "size": 3}
        config["path"] = output_folder
        print(f"Configuration for DefaultChunkingConnection: {config}")

        default_connection = DefaultChunkingConnection(**config)
        chunks = default_connection.chunk()

    except Exception as e:
        print(f"An error occurred: {e}")
