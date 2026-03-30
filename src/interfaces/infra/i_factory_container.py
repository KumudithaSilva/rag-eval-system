from typing import Protocol


class IFactoryContainer(Protocol):
    """
    Contract for any factory container that wires services.
    """

    def create_llm_connection_service(self):
        """
        Create and return a service that provides an LLM client.

        Returns:
            ChatConnectionService instance or compatible object.
        """
        pass

    def create_prompt_generation_service(self):
        """
        Create and return a prompt generation service.

        Returns:
            PromptGenerationService instance or compatible object.
        """
        pass
