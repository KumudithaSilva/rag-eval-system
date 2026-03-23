from abc import ABC, abstractmethod
import pandas as pd


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
    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch all documents from the database.

        Returns:
            pd.DataFrame: DataFrame containing all documents.
        """
        pass
