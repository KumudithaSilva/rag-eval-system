import os
from typing import IO, Union
from interfaces.infra.i_file_extractor import IFileExtractor
from interfaces.infra.i_knowledge_base import IKnowledgeBaseService


class KnowledgeBaseGenerationService(IKnowledgeBaseService):
    """
    Service responsible for generating knowledge base folders from input files.
    """

    def __init__(self, file_extractor: IFileExtractor):
        """
        Initialize the KnowledgeBaseGenerationService with a file extractor.

        Args:
            file_extractor (IFileExtractor): An instance of a file extractor to process input files

        """
        self.file_extractor: IFileExtractor = file_extractor

    def generate(self, source: Union[str, IO[bytes]], folder_name: str) -> str:
        """
        Genereate knowledge base folders and return path to the generated folder.

        Args:
            source (Union[str, IO[bytes]]): File path or file-like object containing the archive.
            folder_name (str): Name of the folder

        Returns:
            str: Path to the generated folder.
        """
        folder_name = os.path.splitext(folder_name)[0]

        self.file_extractor.extract_file(source=source)
        base_folder = self.file_extractor.extract_file_to_folder(folder=folder_name)

        return base_folder
