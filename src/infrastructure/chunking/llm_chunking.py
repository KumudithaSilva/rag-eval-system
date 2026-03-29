from typing import Dict, List
from interfaces.chunking.i_chunking_strategy import IChunkingStrategy
from interfaces.llm.i_ai_client import IAIClient
from logs.logger_singleton import Logger


class LLMChunking(IChunkingStrategy):
    """
    Chunk text using a OpenRouter with LLM.

    Attributes:
        chat_messages (List[Dict]): List of formatted chat messages.
        llm_client (AIClient): Client interface for interacting with the LLM.
    """

    def __init__(self, message: List[Dict], llm_client=IAIClient, logger=None):
        """
        Initialize the LLMChunking strategy.

        Args:
            chat_messages (List[Dict]): List of formatted chat messages.
            llm_client (AIClient): Client used to communicate with the LLM.
            logger (Logger, optional): A logger instance. If None, a default
                logger is created using the class name.
        """
        self.chat_messages: List[Dict] = message
        self.llm_client = llm_client
        self.logger = logger or Logger(self.__class__.__name__)

    def chunk(self) -> List:
        """
        Chunk the given text using the LLM.

        Returns:
            List: List of chunked documents.
        """
        message = self.chat_messages
        text_splitter = self.llm_client.chat_completions_create(
            messages=message, structured=True
        )
        return text_splitter


# if __name__ == "__main__":
#     from infrastructure.infra.open_router_provider import OpenRouterProvider
#     from infrastructure.infra.env_loader import DotEnvLoader
#     from infrastructure.infra.open_router_client import OpenRouterAIClientWrapper
#     from components.prompt_generation import PromptGenerationService
#     from infrastructure.infra.file_extractor import FileExtractor
#     from infrastructure.infra.chunking_prompt import PromptProvider
#     from utils.document_loader import doc_convert

#     extractor = FileExtractor()
#     prompt_provider = PromptProvider()
#     prompt_generater = PromptGenerationService(prompt_provider=prompt_provider)

#     env_loader = DotEnvLoader()
#     key_provider = OpenRouterProvider(env_loader=env_loader)
#     llm_client = OpenRouterAIClientWrapper(key_provider, model="openai/gpt-4o")

#     test_archive_path = "knowledge-base.zip"

#     try:
#         extracted = extractor.extract_file(test_archive_path)
#         print(f"Extracted {len(extracted)} files into memory.")

#         output_folder = extractor.extract_file_to_folder()
#         print(f"Files saved to: {output_folder}")

#         documents = doc_convert(output_folder) if output_folder else []
#         test_document = documents[0]

#         prompt_generaterate = prompt_generater.generate(test_document)

#         LLMChunking_instance = LLMChunking(
#             message=prompt_generaterate, llm_client=llm_client
#         )

#         chunks = LLMChunking_instance.chunk()
#         print(chunks)

#     except Exception as e:
#         print(f"Error during file extraction: {e}")
