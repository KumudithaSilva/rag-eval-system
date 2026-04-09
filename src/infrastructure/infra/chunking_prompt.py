from typing import List

from config.constants import AVERAGE_CHUNK_SIZE
from dto.document_data import DocumentData
from interfaces.chat.i_oneshot_prompt import IPrompt
from logs.logger_singleton import Logger
from langchain_core.documents import Document


class PromptProvider(IPrompt):
    """
    Provider system prompts for one-shot learning tasks.
    """

    def __init__(self, logger=None):
        """
        Initialize the PromptProvider instance.

        Args:
            logger (Logger, optional): A logger instance. If None, a
                default logger is created using the class name.
        """
        self.logger = logger or Logger(self.__class__.__name__)

    def system_prompt(self) -> str:
        """
        Get the system prompt.

        Returns:
            str: The system prompt string.
        """
        system_prompt = """You take a document and you split the document into overlapping chunks for a KnowledgeBase."""
        return system_prompt

    def user_prompt(self, document: Document) -> List:
        """
        Get the user prompt.

        Args:
            document (Document): The LangChain source document to be chunked.

        Returns:
            List: The user prompt string.
        """
        doc_data = DocumentData(
            page_content=document.page_content, metadata=document.metadata
        )

        how_many = (len(document.page_content) // AVERAGE_CHUNK_SIZE) + 1
        self.logger.info(f"Calculated number of chunks: {how_many}")

        user_prompt = f"""
        Below is the document that needs chunking.

        The document is of type: {doc_data.doc_type}
        The document has been retrieved from: {doc_data.source}

        You should divide up the document as you see fit, being sure that the entire document is returned in the chunks - don't leave anything out.
        This document should probably be split into {how_many} chunks, but you can have more or less as appropriate.
        There should be overlap between the chunks as appropriate; typically about 25% overlap or about 50 words, so you have the same text in multiple chunks for best retrieval results.

        For each chunk, you should provide a headline, a summary, and the original text of the chunk.
        Together your chunks should represent the entire document with overlap.

        Here is the document:

        {doc_data.page_content}

        Respond with the chunks.
        """
        return user_prompt
