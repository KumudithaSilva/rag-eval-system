from typing import Dict, List
from interfaces.chat.i_oneshot_prompt import IPrompt
from interfaces.chat.i_prompt_generation import IPromptGenereateService
from logs.logger_singleton import Logger
from langchain_core.documents import Document


class PromptGenerationService(IPromptGenereateService):
    """
    Initializes Service that responsible for generating chat prompts for an LLM.

    Attributes:
        prompt_provider (IPrompt): Instance responsible for providing
            system and user prompts.
        chat_messages (List[Dict]): List of formatted chat messages.
        logger (Logger): Logger instance for logging operations.
    """

    def __init__(self, prompt_provider: IPrompt, logger=None):
        """
        Initialize the PromptGenerationService.

        Args:
            prompt_provider (IPrompt): An implementation that provides
                system and user prompt content.
            logger (Logger, optional): Logger instance. If not provided,
                a default logger is created using the class name.

        """
        self.prompt_provider = prompt_provider
        self.chat_messages: List[Dict] = []
        self.logger = logger or Logger(self.__class__.__name__)

    def generate(self, document: Document) -> List:
        """
        Generate a prompt for the client.

        Args:
            document (Document): The LangChain source document to be chunked.

        Returns:
            List: Return combined system and user prompt with documents.
        """
        self.chat_messages = [
            {"role": "system", "content": self.prompt_provider.system_prompt()},
            {"role": "user", "content": self.prompt_provider.user_prompt(document)},
        ]
        return self.chat_messages

    def generate_batch(self, documents: List[Document]) -> List[List[Dict]]:
        """
        Generate prompts for a batch of documents.

        Args:
            documents (List[Document]): A list of LangChain source documents to be chunked.

        Returns:
            List[List[Dict]]: A list of chat message batches, where each batch corresponds to a document.
        """
        batch_messages = []
        for document in documents:
            messages = self.generate(document)
            batch_messages.append(messages)
        return batch_messages
