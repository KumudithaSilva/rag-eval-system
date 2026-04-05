from typing import Dict

from interfaces.db.i_mongo_connection import IMongoConnection
from interfaces.infra.i_pipeline import IPipelineStep
from interfaces.infra.i_vector_store import IVectorStoreService
from logs.logger_singleton import Logger


class MongoStoreStep(IPipelineStep):
    """
    Pipeline step for mongo store operations.

    args:
        mongo_store_service (IMongoConnection): The mongo store service to use for storing configuration details.
        config_details (Dict): Configuration details for RAG eval system.
    """

    def __init__(
        self, mongo_store_service: IMongoConnection, config_details: Dict, logger=None
    ):
        """
        Initialize MongoConnectionService.

        Args:
            mongo_store_service (IMongoConnection): The mongo store service to use for storing configuration details.
            config_details (Dict): Configuration details for RAG eval system.
            logger (Logger, optional): A logger instance. If None, a default
                logger is created using the class name.
        """
        self.mongo_store_service = mongo_store_service
        self.config_details = config_details
        self.logger = logger or Logger(self.__class__.__name__)

    def run(self, data: dict):
        """
        Run the mongo storing step on the input data.

        args:
            data (dict): Store the results of the mertic evaluation process.
        """
        mongo_store_service = self.mongo_store_service.connect()

        combined_data = {**self.config_details, **data["evaluator"]}
        self.logger.debug(f"Updated Configuration {combined_data}")

        mongo_store_service.insert_data(combined_data)

        self.logger.info("MongoStoreStep completed successfully")
        self.logger.debug(f"Data Store {combined_data}")

        return data
