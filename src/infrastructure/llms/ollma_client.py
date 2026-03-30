from typing import Dict, List
from dto.llm_schemas import Chunks
from interfaces.llm.i_ai_client import IAIClient
from langchain_ollama import ChatOllama
from utils.ollama import ensure_ollama_running


class OllamaClientWrapper(IAIClient):
    """
    Concrete wrapper for the Ollama Python library with optional structured output.

    Attributes:
        base_client (ChatOllama): Base ChatOllama client instance.
    """

    def __init__(self, model: str = "llama3.2"):
        """
        Initialize Ollama client wrapper.

        Args:
            model: Model name to use.
        """
        ensure_ollama_running()
        self.base_client = ChatOllama(model=model, base_url="http://localhost:11434")

    def chat_completions_create(
        self, messages: List[Dict], structured: bool = True
    ) -> str:
        """
        Sends a chat completion request to the AI backend.

        Args:
            messages: List of combined system and user prompt with documents embedded.
            structured: If True, returns structured output; otherwise plain text.

        Returns:
             The AI-generated response, parsed into the structured schema (Chunks).
        """
        if structured:
            self.client = self.base_client.with_structured_output(
                Chunks, method="json_schema"
            )
        else:
            self.client = self.base_client
        response = self.client.invoke(messages)

        return response
