from dataclasses import dataclass
from typing import Dict


@dataclass
class DocumentData:
    page_content: str
    metadata: Dict[str, str]

    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")

    @property
    def doc_type(self) -> str:
        return self.metadata.get("doc_type", "unknown")
