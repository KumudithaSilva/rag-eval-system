from abc import ABC, abstractmethod
from interfaces.db.i_database_interface import IDatabaseRepository


class IMongoConnection(ABC):
    """
    Interface for initializing mongo connection.
    """

    @abstractmethod
    def connect(self) -> IDatabaseRepository:
        """
        Initializing Mongo Connection.

        Returns:
            IDatabaseRepository: Abstract repository interface
        """
        pass
