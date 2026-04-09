from abc import ABC, abstractmethod
from typing import IO, Union


class IKnowledgeBaseService(ABC):
    """
    Interface for knowledge base service, which is responsible for generating folder structure.
    """

    @abstractmethod
    def generate(self, path: Union[str, IO[bytes]]) -> str:
        """
        Genereate knowledge base folders and return path to the generated folder.

        Args:
            path (str | IO[bytes]): Path to file or file-like object.

        Returns:
            str: Path to the generated folder.
        """
        pass
