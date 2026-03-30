from container.factory_container import FactoryContainer
from infrastructure.chunking.llm_chunking import LLMChunking
from interfaces.chunking.i_chunking_strategy import IChunkingStrategy
from interfaces.infra.i_factory_container import IFactoryContainer
from utils.document_loader import doc_convert


class LLMChunkingFactory:
    """
    Factory responsible for creating LLMChunking instances.
    """

    def __init__(self, factory_container: IFactoryContainer | None = None):
        self.factory_container: IFactoryContainer = (
            factory_container or FactoryContainer()
        )

    def create(self, config: dict) -> IChunkingStrategy:
        # Extract configuration parameters
        path = config.get("path")
        model = config.get("model")

        # Load documents
        documents = doc_convert(path) if path else []
        documents = documents[:1]

        # List of prompts
        prompts_service = self.factory_container.create_prompt_generation_service()
        prompts = prompts_service.generate_batch(documents)

        # LLM Client
        llm_client_service = self.factory_container.create_llm_connection_service()
        llm_client = llm_client_service.connect(model=model)

        # Create strategy
        return LLMChunking(messages=prompts, llm_client=llm_client)


# if __name__ == "__main__":
#     from infrastructure.infra.file_extractor import FileExtractor

#     extractor = FileExtractor()

#     test_archive_path = "knowledge-base.zip"

#     try:
#         extracted = extractor.extract_file(test_archive_path)
#         output_folder = extractor.extract_file_to_folder()

#         config = {"model": "llama3.2"}
#         config["path"] = output_folder

#         llm_connection = LLMChunkingFactory()
#         chunks = llm_connection.create(config).chunk()

#         print(f"Generated {len(chunks)} chunks from the documents.")
#         print(chunks[0])

#     except Exception as e:
#         print(f"An error occurred: {e}")
