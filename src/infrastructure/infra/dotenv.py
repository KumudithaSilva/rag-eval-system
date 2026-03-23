from dotenv import find_dotenv, load_dotenv

from interfaces.infra.i_env_loader import IEnvLoader
from logs.logger_singleton import Logger


class DotEnvLoader(IEnvLoader):
    """
    Loader for environment variables from a .env file.
    """

    def __init__(self, logger=None):
        """
        Initialize the DotEnvLoader instance.

        Args:
            logger (Logger, optional): A logger instance. If None, a default
                logger is created using the class name.
        """
        self.logger = logger or Logger(self.__class__.__name__)

    def load_env_variables(self) -> None:
        """
        Load environment variables from a `.env` file if present.

        Returns:
            None

        Raises:
            Exception: Catches any exception raised during loading and logs it.
        """
        try:
            dotenv_path = find_dotenv()
            if dotenv_path:
                load_dotenv(dotenv_path)
                self.logger.info(f".env file found and loaded from: {dotenv_path}")
            else:
                self.logger.warning(
                    ".env file not found; using existing environment variables"
                )
        except ImportError:
            self.logger.warning(
                "python-dotenv not installed; using existing environment variables"
            )
        except Exception as e:
            self.logger.error(f"Error loading .env file: {e}")
