from fastapi import Request
from logs.logger_singleton import Logger

logger = Logger(name="fastapi-dependencies")


def get_mongo_client(request: Request):
    client = request.app.state.mongo_client
    if client is None:
        logger.error("Mongo client not initialized")
        raise RuntimeError("Mongo client not initialized")
    return client
