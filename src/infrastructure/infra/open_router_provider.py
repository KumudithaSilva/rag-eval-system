import os

from interfaces.infra.i_api_key_provider import IApiKeyProvider
from interfaces.infra.i_env_loader import IEnvLoader
from logs.logger_singleton import Logger


class OpenRouterProvider(IApiKeyProvider):
    """
    Provider for OpenRouter, loading from environment variables.
    """

    def __init__(self, env_loader: IEnvLoader, logger=None):
        """
        Initialize the OpenRouterProvider instance.

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
        Load and validate the OpenRouter from environment variables.

        Returns:
            str: The valid OpenRouter Key.

        Raises:
            EnvironmentError: If the Key is missing or invalid.
        """
        # Load environment variables
        self.env_loader.load_env_variables()

        # Fetch the KEY
        api_key = os.getenv("OPEN_ROUTER")

        if not api_key:
            self.logger.error("Error: OPENROUTER not set")
            raise EnvironmentError("OPENROUTER not set")

        if api_key.startswith("sk-or-v1"):
            self.logger.info("OPENROUTE Key found and validated")
            return api_key

        self.logger.error("Invalid OPENROUTER key format")
        raise EnvironmentError("Invalid OPENROUTER key format")
