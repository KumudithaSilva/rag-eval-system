from fastapi import Request
from container.rag_container import RagEvalContainer
from logs.logger_singleton import Logger

logger = Logger(name="fastapi-dependencies")


def get_rag_container(request: Request) -> RagEvalContainer:
    container = request.app.state.rag_container
    if container is None:
        raise RuntimeError("RAG container not initialized")
    return container


def get_mongo_client(request: Request):
    client = request.app.state.mongo_client
    if client is None:
        logger.error("Mongo client not initialized")
        raise RuntimeError("Mongo client not initialized")
    return client
