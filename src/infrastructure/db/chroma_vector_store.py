import os
from typing import List, Optional

from langchain_core.documents import Document
from langchain_chroma import Chroma

from infrastructure.logs.logger_singleton import Logger
from interfaces.embedding.i_embedding import IEmbeddingModel
from interfaces.infra.i_vector_store import IVectorStoreService


class VectorStoreService(IVectorStoreService):
    """
    Service for managing vector store creation.
    """

    def __init__(
        self,
        embedding_model: IEmbeddingModel,
        logger=None,
    ):
        """
        Initialize the vector store service.

        Args:
            embedding_model (IEmbeddingModel): Embedding model instance.
            logger (Logger, optional): Logger instance.
        """
        self._embedding_model = embedding_model
        self._logger = logger or Logger(self.__class__.__name__)

    def create(
        self,
        chunks: List[Document],
        persist_directory: str = "vector_store",
    ) -> Optional[Chroma]:
        """
        Create a Chroma vector store from document chunks.

        Args:
            chunks (List[Document]): Document chunks.
            persist_directory (str, optional): Directory to persist the vector store.

        Returns:
            Optional[Chroma]: The created vector store.
        """
        if not chunks:
            self._logger.warning("No chunks provided.")
            return None

        embedding = self._embedding_model.get_model()

        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [str(index) for index in range(len(chunks))]

        try:
            if os.path.exists(persist_directory):
                self._logger.info("Deleting existing vector store...")
                Chroma(
                    persist_directory=persist_directory,
                    embedding_function=embedding,
                ).delete_collection()

            vector_store = Chroma.from_texts(
                texts=texts,
                metadatas=metadatas,
                ids=ids,
                embedding=embedding,
                persist_directory=persist_directory,
            )

            self._logger.info(f"Vector store created with {len(texts)} documents.")

            return vector_store

        except Exception as exc:
            self._logger.error(
                f"Failed to create vector store: {exc}",
            )
            raise
