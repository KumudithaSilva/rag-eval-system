import os

from interfaces.infra.i_api_key_provider import IApiKeyProvider
from interfaces.infra.i_env_loader import IEnvLoader
from logs.logger_singleton import Logger


class OpenAIApiKeyProvider(IApiKeyProvider):
    """
    Provider for OpenAI API keys, loading from environment variables.
    """

    def __init__(self, env_loader: IEnvLoader, logger=None):
        """
        Initialize the OpenAIApiKeyProvider instance.

        Args:
            env_loader (IEnvLoader): An environment loader instance used
                to load environment variables.
            logger (Logger, optional): A logger instance. If None, a
                default logger is created using the class name.
        """
        self.env_loader = env_loader
        self.logger = logger or Logger(self.__class__.__name__)

    def get_api_key(self) -> str:
        """
        Load and validate the OpenAI API key from environment variables.

        Returns:
            str: The valid OpenAI API key.

        Raises:
            EnvironmentError: If the API key is missing or invalid.
        """
        # Load environment variables
        self.env_loader.load_env_variables()

        # Fetch the API key
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            self.logger.error("Error: OPENAI_API_KEY not set")
            raise EnvironmentError("OPENAI_API_KEY not set")

        if api_key.startswith("sk-proj-"):
            self.logger.info("OpenAI API key found and validated")
            return api_key

        self.logger.error("Invalid OpenAI API key format")
        raise EnvironmentError("Invalid OpenAI API key format")
