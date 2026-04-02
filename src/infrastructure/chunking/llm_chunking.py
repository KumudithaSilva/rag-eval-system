from typing import Dict, List
from interfaces.chunking.i_chunking_strategy import IChunkingStrategy
from interfaces.llm.i_ai_client import IAIClient
from logs.logger_singleton import Logger
from utils.result_builder import ResultBuilder


class LLMChunking(IChunkingStrategy):
    """
    Chunk text using a OpenRouter with LLM.

    Attributes:
        chat_messages (List[List[Dict]]): A list of chat message batches to be sent to the LLM for chunking.
        llm_client (AIClient): Client interface for interacting with the LLM.
    """

    def __init__(self, messages: List[List[Dict]], llm_client=IAIClient, logger=None):
        """
        Initialize the LLMChunking strategy.

        Args:
            messages (List[List[Dict]]): A list of message batches.
            llm_client (AIClient): Client used to communicate with the LLM.
            logger (Logger, optional): A logger instance. If None, a default
                logger is created using the class name.
        """
        self.chat_messages: List[List[Dict]] = messages
        self.llm_client = llm_client
        self.logger = logger or Logger(self.__class__.__name__)

    def chunk(self) -> List:
        """
        Chunk the given text using the LLM for each message batch.

        Returns:
            List: A list of responses from the LLM, one per message batch.
        """
        for message_batch in self.chat_messages:
            response = self.llm_client.chat_completions_create(
                messages=message_batch, structured=True
            )
            chunk_response = response.chunks
            results = [ResultBuilder.from_chunk(chunk) for chunk in chunk_response]

        self.logger.debug(f"Generated {results[0]} chunks.")
        return results
