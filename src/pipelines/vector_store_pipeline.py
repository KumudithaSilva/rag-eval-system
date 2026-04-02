from interfaces.infra.i_pipeline import IPipelineStep
from interfaces.infra.i_vector_store import IVectorStoreService


class VectorStoreStep(IPipelineStep):
    """
    Pipeline step for vector store operations.

    args:
        vector_store_service (IVectorStoreService): The vector store service to use for storing and retrieving vectors.
    """

    def __init__(self, vector_store_service: IVectorStoreService):
        """
        Initialize ChunkingStep with a chunking strategy.

        Args:
            vector_store_service (IVectorStoreService): The vector store service to use for storing and retrieving vectors.
        """
        self.vector_store = vector_store_service

    def run(self, data: dict):
        """
        Run the chunking step on the input data.

        args:
            data (dict): Store the results of the chunking process.
        """
        vector_store_service = self.vector_store.create(chunks=data["chunks"])
        count = vector_store_service._collection.count()
        data["count"] = count
        return data
