from logging import Logger

from pymongo import MongoClient
import pandas as pd
from interfaces.db.i_database_interface import IDatabaseRepository
from interfaces.infra.i_api_key_provider import IApiKeyProvider


class MongoRepository(IDatabaseRepository):
    """
    Concrete implementation of MongoInterface using PyMongo.
    """

    def __init__(
        self,
        key_provider: IApiKeyProvider,
        db_name: str = "rag_db",
        collection_name: str = "rag_eval_2",
        logger=None,
    ):
        """
        Initialize MongoDB connection and collection.

        Args:
            key_provider (IApiKeyProvider): Interface to obtain OpenAI API key.
            db_name (str): Name of the database.
            collection_name (str): Name of the collection.
            logger (Logger, optional): A logger instance. If None, default logger is created using the class name.
        """
        self.key_provider = key_provider.get_api_key()
        self.db_name = db_name
        self.collection_name = collection_name

        self.client = None
        self.collection = None

        self.logger = logger or Logger(self.__class__.__name__)

    def _connect(self) -> None:
        """
        Establish a connection to the MongoDB database if not already connected.

        Raises:
            Exception: If the connection to MongoDB fails.
        """
        if self.client is None:
            mongo_url = self.key_provider

            self.client = MongoClient(
                mongo_url,
                tls=True,
                tlsAllowInvalidCertificates=False,
            )

            self.collection = self.client[self.db_name][self.collection_name]

    def insert_data(self, data: dict) -> str | None:
        """
        Insert a document into the MongoDB collection.

        Args:
            data (dict): Dictionary to insert.

        Returns:
            str | None: Inserted document ID as a string if successful,
            otherwise None.
        """
        try:
            self._connect()
            result = self.collection.insert_one(data)
            return str(result.inserted_id)
        except Exception as exc:
            print(f"Insert error: {exc}")
            return None

    def fetch_data(self) -> pd.DataFrame:
        """
        Retrieve all documents and convert them into a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing all records. Returns an empty
            DataFrame if no data is found or an error occurs.
        """
        try:
            self._connect()
            documents = list(self.collection.find())

            if not documents:
                return pd.DataFrame()

            df = pd.json_normalize(documents, sep=".")
            df = df.drop(columns=["_id", "collection_name"], errors="ignore")

            return df

        except Exception as exc:
            print(f"Fetch error: {exc}")
            return pd.DataFrame()
