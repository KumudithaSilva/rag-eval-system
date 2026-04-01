from infrastructure.chunking.llm_chunking import LLMChunking
from interfaces.chat.i_chatbot_connection import IChatConnection
from interfaces.chat.i_prompt_generation import IPromptGenereateService
from interfaces.chunking.i_chunking_strategy import IChunkingStrategy
from interfaces.infra.i_chunking_factory import IChunkingFactory
from interfaces.infra.i_factory_container import IFactoryContainer
from utils.document_loader import doc_convert


class LLMChunkingFactory(IChunkingFactory):
    """
    Factory for creating LLMChunking strategy instances.
    """

    def __init__(
        self,
        prompt_service: IPromptGenereateService,
        llm_client: IChatConnection,
    ):
        self.prompt_service = prompt_service
        self.llm_client = llm_client

    def create(self, config: dict, documents: list) -> IChunkingStrategy:
        """
        Create LLMChunking using preloaded documents.
        """
        # Preprocess documents and generate prompts
        prompts = self.prompt_service.generate_batch(documents)

        # Connect LLM client
        llm_client = self.llm_client.connect(model=config.get("llm_model"))

        # Return strategy instance
        return LLMChunking(messages=prompts, llm_client=llm_client)
