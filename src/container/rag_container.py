from components.mongo_connection import MongoConnectionService
from interfaces.infra.i_api_key_provider import IApiKeyProvider
from interfaces.infra.i_env_loader import IEnvLoader
from infrastructure.infra.mongo_url_provider import MongoUrlProvider
from infrastructure.infra.dotenv import DotEnvLoader


class RagEvalContainer:
    """
    Factory to wire all dependencies and return orchestrator service instances.
    """

    def create_mongo_connection_service(
        self,
        env_loader: IEnvLoader | None = None,
        key_provider: IApiKeyProvider | None = None,
    ):
        """
        Create and return a Mongo connection service.

        Args:
            env_loader (IEnvLoader, optional): Environment loader.
            key_provider (IApiKeyProvider, optional): Mongo URL provider.

        Returns:
            IMongoConnection: Mongo connection service instance.
        """
        if env_loader is None:
            env_loader = DotEnvLoader()

        if key_provider is None:
            key_provider = MongoUrlProvider(env_loader)

        return MongoConnectionService(key_provider)


if __name__ == "__main__":
    import container

    container = RagEvalContainer()

    connection_service = container.create_mongo_connection_service()
    mongo_client = connection_service.connect()

    if mongo_client:
        print(mongo_client.fetch_data())
