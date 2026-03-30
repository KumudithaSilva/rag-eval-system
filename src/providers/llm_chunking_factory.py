from container.factory_container import FactoryContainer
from infrastructure.chunking.llm_chunking import LLMChunking
from interfaces.chunking.i_chunking_strategy import IChunkingStrategy
from interfaces.infra.i_chunking_factory import IChunkingFactory
from interfaces.infra.i_factory_container import IFactoryContainer
from utils.document_loader import doc_convert


class LLMChunkingFactory(IChunkingFactory):
    """
    Factory responsible for creating LLMChunking instances.
    """

    def __init__(self, factory_container: IFactoryContainer | None = None):
        self.factory_container: IFactoryContainer = (
            factory_container or FactoryContainer()
        )

    def create(self, config: dict, path: str) -> IChunkingStrategy:
        # Add path for document loading
        path = path
        # Extract configuration parameters
        model = config.get("llm_model")

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
