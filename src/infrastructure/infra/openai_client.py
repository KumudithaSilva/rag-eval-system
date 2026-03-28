from typing import Dict, List

from dto.llm_schemas import Chunks
from interfaces.infra.i_api_key_provider import IApiKeyProvider
from interfaces.llm.i_ai_client import IAIClient
from langchain_openrouter import ChatOpenRouter


class OpenRouterAIClientWrapper(IAIClient):
    """
    Concrete wrapper for the OpenRouter Python library with optional structured output.

    Attributes:
        api_key (str): API key obtained from IApiKeyProvider.
        base_client (ChatOpenRouter): Base OpenRouter client instance.
    """

    def __init__(self, key_provider: IApiKeyProvider, model: str):
        """
        Initialize OpenRouter client wrapper.

        Args:
            key_provider (IApiKeyProvider): Interface to obtain OpenAI API key.
            model: Model name to use.
        """
        self.api_key = key_provider.get_api_key()
        self.base_client = ChatOpenRouter(
            api_key=self.api_key, model=model, app_title=None
        )

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


if __name__ == "__main__":
    from infrastructure.infra.open_router_provider import OpenRouterProvider
    from infrastructure.infra.env_loader import DotEnvLoader

    env_loader = DotEnvLoader()
    key_provider = OpenRouterProvider(env_loader=env_loader)

    open_router = OpenRouterAIClientWrapper(
        key_provider=key_provider, model="openai/gpt-4o"
    )

    message = [
        (
            "system",
            "You are a helpful assistant that translates English to French. Translate the user sentence.",
        ),
        ("human", "I love programming."),
    ]
    open_router_message = open_router.chat_completions_create(
        messages=message, structured=False
    )
    print(open_router_message)
