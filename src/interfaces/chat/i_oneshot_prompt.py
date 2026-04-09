from abc import ABC, abstractmethod
from typing import List

from dto.document_data import DocumentData


class IPrompt(ABC):

    @abstractmethod
    def system_prompt(self) -> str:
        pass

    @abstractmethod
    def user_prompt(document: DocumentData) -> List:
        pass
