from abc import ABC, abstractmethod
from typing import List, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma


class IVectorStoreService(ABC):
    """
    Interface for vector store services.
    """

    @abstractmethod
    def create(
        self,
        chunks: List[Document],
        persist_directory: str = "vector_store",
    ) -> Optional[Chroma]:
        """
        Create a vector store from document chunks.

        Args:
            chunks (List[Document]): Document chunks.
            persist_directory (str, optional): Directory to persist the vector store.

        Returns:
            Optional[Chroma]: Created vector store or None.
        """
        pass
