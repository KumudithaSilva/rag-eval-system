import logging

from core.singleton_meta import SingletonMeta
from interfaces.logging.i_logger_interface import ILogger


class Logger(ILogger, metaclass=SingletonMeta):
    """
    Logger singleton per name.

    Provides a single logger instance per class name or custom name.
    Supports both console and file logging with standard formatting.
    """

    def __init__(self, name: str = "Singleton_Logger"):
        # Create or get a logger with the specified name
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.DEBUG)  # Capture all levels

        # Only add handlers once to avoid duplicates
        if not self._logger.hasHandlers():
            # logs all DEBUG and messages to a file
            file_handler = logging.FileHandler("logs.log")
            file_handler.setLevel(logging.DEBUG)

            # logs INFO and messages to the console
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            # Define log message format
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s.%(funcName)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            # Add handlers to logger
            self._logger.addHandler(file_handler)
            self._logger.addHandler(console_handler)

    # Logging methods with stacklevel 2 to show correct caller
    def debug(self, message: str):
        self._logger.debug(message, stacklevel=2)

    def info(self, message: str):
        self._logger.info(message, stacklevel=2)

    def warning(self, message: str):
        self._logger.warning(message, stacklevel=2)

    def error(self, message: str):
        self._logger.error(message, stacklevel=2)

    def exception(self, message: str):
        self._logger.exception(message, stacklevel=2)

    def critical(self, message: str):
        self._logger.critical(message, stacklevel=2)
