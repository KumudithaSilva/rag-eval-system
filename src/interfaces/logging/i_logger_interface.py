from abc import ABC, abstractmethod


class ILogger(ABC):
    """
    Logger interface defining standard logging methods.
    Every concrete logger must implement these methods.
    """

    @abstractmethod
    def debug(self, message: str) -> None:
        pass

    @abstractmethod
    def info(self, message: str) -> None:
        pass

    @abstractmethod
    def warning(self, message: str) -> None:
        pass

    @abstractmethod
    def error(self, message: str) -> None:
        pass

    @abstractmethod
    def exception(self, message: str) -> None:
        pass

    @abstractmethod
    def critical(self, message: str) -> None:
        pass
