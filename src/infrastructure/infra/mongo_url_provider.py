import os

from interfaces.infra.i_api_key_provider import IApiKeyProvider
from interfaces.infra.i_env_loader import IEnvLoader
from logs.logger_singleton import Logger


class MongoUrlProvider(IApiKeyProvider):
    """
    Provider for Mongo URL, loading from environment variables.
    """

    def __init__(self, env_loader: IEnvLoader, logger=None):
        """
        Initialize the MongoUrlProvider instance.

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
        Load and validate the Mongo URL from environment variables.

        Returns:
            str: The valid Mongo URL.

        Raises:
            EnvironmentError: If the URL is missing or invalid.
        """
        # Load environment variables
        self.env_loader.load_env_variables()

        # Fetch the URL
        api_key = os.getenv("MONGO_URI")

        if not api_key:
            self.logger.error("Error: MONGO_URI not set")
            raise EnvironmentError("MONGO_URI not set")

        if api_key.startswith("mongodb+srv"):
            self.logger.info("MONGO URI found and validated")
            return api_key

        self.logger.error("Invalid MONGO URI key format")
        raise EnvironmentError("Invalid MONGO URI key format")
