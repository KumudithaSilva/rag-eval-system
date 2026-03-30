from infrastructure.chunking.default_chunking import DefaultChunking
from interfaces.chunking.i_chunking_strategy import IChunkingStrategy
from utils.document_loader import doc_convert


class DefaultChunkingFactory:
    """
    Factory responsible for creating DefaultChunking instances.
    """

    def create(self, config: dict) -> IChunkingStrategy:
        # Extract configuration parameters
        path = config.get("path")

        # Load documents
        documents = doc_convert(path) if path else []

        # Remove path from config before passing
        clean_config = {k: v for k, v in config.items() if k != "path"}

        # Create strategy
        return DefaultChunking(documents=documents, **clean_config)


# if __name__ == "__main__":
#     from infrastructure.infra.file_extractor import FileExtractor

#     extractor = FileExtractor()

#     test_archive_path = "knowledge-base.zip"

#     try:
#         extracted = extractor.extract_file(test_archive_path)
#         output_folder = extractor.extract_file_to_folder()

#         config = {"k": 2, "size": 3}
#         config["path"] = output_folder

#         default_connection = DefaultChunkingFactory()
#         chunks = default_connection.create(config).chunk()

#         print(f"Generated {len(chunks)} chunks from the documents.")
#         print(chunks[0])

#     except Exception as e:
#         print(f"An error occurred: {e}")
