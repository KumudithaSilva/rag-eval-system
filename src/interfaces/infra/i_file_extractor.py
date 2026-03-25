from abc import ABC, abstractmethod
from typing import IO, Dict, Union


class IFileExtractor(ABC):

    @abstractmethod
    def extract_file(self, path: Union[str, IO[bytes]]) -> Dict[str, bytes]:
        pass

    @abstractmethod
    def extract_file_to_folder(self, folder: str = "rag_files") -> str:
        pass
