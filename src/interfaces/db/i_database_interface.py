from abc import ABC, abstractmethod


class IDatabaseRepository(ABC):
    """
    Abstract base class that defines the contract for Database operations.
    """

    @abstractmethod
    def insert_data(self, data: dict) -> str:
        """
        Insert a single document into the database.

        Args:
            data (dict): Dictionary representing the document.

        Returns:
            str: Inserted document ID as a string.
        """
        pass

    @abstractmethod
    def fetch_data(self) -> list[dict]:
        """
        Fetch all documents from the database.

        Returns:
            list[dict]: List of records retrieved from the database
        """
        pass
