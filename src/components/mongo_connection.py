from infrastructure.db.mongo_repository import MongoRepository
from interfaces.db.i_mongo_connection import IMongoConnection
from interfaces.db.i_database_interface import IDatabaseRepository
from interfaces.infra.i_api_key_provider import IApiKeyProvider


class MongoConnectionService(IMongoConnection):
    """
    Initializes Mongo connection using interface-based dependencies.
    """

    def __init__(self, key_provider: IApiKeyProvider):
        """
        Initialize MongoConnectionService.

        Args:
            key_provider (IApiKeyProvider): Interface to get mongo url.
        """
        self.key_provider: IApiKeyProvider = key_provider
        self.client: IDatabaseRepository | None = None

    def connect(self) -> IDatabaseRepository:
        """
        Create and return the Mongo client. Only creates it once per instance.

        Returns:
            IDatabaseRepository: Database repository instance.
        """
        if self.client is None:
            self.client = MongoRepository(self.key_provider)
        return self.client
