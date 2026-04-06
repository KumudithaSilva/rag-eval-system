from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import Dict, List
from interfaces.chunking.i_chunking_strategy import IChunkingStrategy
from interfaces.llm.i_ai_client import IAIClient
from logs.logger_singleton import Logger
from utils.result_builder import ResultBuilder


class LLMChunking(IChunkingStrategy):
    """
    Chunk text using an LLM with parallel processing.

    Note:
        Uses threads to process multiple message batches concurrently. Threads
        share the same LLM client instance, which makes it safe and faster for I/O-bound operations (like LLM API calls).

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

    def _process_batch(self, batch):
        """
        Process a single batch of messages using the LLM.

        Args:
            batch: One batch of messages.

        Returns:
            List of chunked results from the LLM.
        """
        response = self.llm_client.chat_completions_create(
            messages=batch, structured=True
        )
        return [ResultBuilder.from_chunk(chunk) for chunk in response.chunks]

    def chunk(self) -> List:
        """
        Chunk all message batches using threads.

        Returns:
            List of all chunked results from all batches.
        """
        start_time = time.time()
        results = []

        # Create a thread pool to process batches concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:

            # Submit each batch to the thread pool
            futures = [
                executor.submit(self._process_batch, batch)
                for batch in self.chat_messages
            ]

            # As each thread completes, gather its results
            for future in as_completed(futures):
                results.extend(future.result())

        end_time = time.time()
        self.logger.info(
            f"ThreadPool chunking took {end_time - start_time:.2f} seconds"
        )
        self.logger.debug(f"Generated {len(results)} chunks.")
        return results
