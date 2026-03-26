import os
import tempfile
from typing import IO, Dict, Union
import zipfile
import rarfile

from interfaces.infra.i_file_extractor import IFileExtractor
from utils.archive_utils import extract_zip, extract_rar
from logs.logger_singleton import Logger


class FileExtractor(IFileExtractor):
    """
    Extract files from supported archive formats (ZIP and RAR).

    Attributes:
        extracted_files (Dict[str, bytes]): Stores extracted file names and their binary content.
    """

    def __init__(self, logger=None):
        """
        Initialize the FileExtractor instance.

        Sets up an empty dictionary to store extracted files in memory.

        Args:
            logger (Logger, optional): A logger instance. If None, a default
                logger is created using the class name.
        """
        self.extracted_files: Dict[str, bytes] = {}
        self.logger = logger or Logger(self.__class__.__name__)

    def extract_file(self, source: Union[str, IO[bytes]]) -> Dict[str, bytes]:
        """
        Extract files from a ZIP or RAR archive.

        Args:
            source (Union[str, IO[bytes]]): File path or file-like object containing the archive.

        Returns:
            Dict[str, bytes]: A dictionary where keys are file names and values are file contents in bytes.

        Raises:
            ValueError: If the file is not a supported archive type (ZIP or RAR).
        """
        temp_path = None

        try:
            # Path is string
            if isinstance(source, str):
                path = source
            else:
                # File-like object, convert to temp file
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    # Read all bytes from stream and write to temp file
                    tmp.write(source.read())
                    path = tmp.name
                    temp_path = path

            # Extract depending on type
            if zipfile.is_zipfile(path):
                self.extracted_files = extract_zip(path)
            elif rarfile.is_rarfile(path):
                self.extracted_files = extract_rar(path)

            self.logger.info("File extracted and saved in temp memory.")
            return self.extracted_files

        except:
            self.logger.error("Unsupported archive type.")
            raise ValueError("Unsupported archive type. Only ZIP or RAR allowed.")

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
                self.logger.info("Temporary data removed")

    def extract_file_to_folder(self, folder: str = "rag_files") -> str:
        """
        Save extracted files from memory to a specified folder.

        Args:
        folder (str | None): Target folder path. Defaults to current working directory.

        Returns:
            str: Absolute path to the folder where files were saved.

        Raises:
            ValueError: If no files have been extracted.
        """
        if not self.extracted_files:
            self.logger.error("No extracted files in memory.")
            raise ValueError("No extracted files in memory.")

        folder = os.path.abspath(folder)

        os.makedirs(folder, exist_ok=True)
        self.logger.info(f"Created rag folder: {folder}")

        for file_name, content in self.extracted_files.items():
            file_path = os.path.join(folder, file_name)

            if file_name.endswith("/"):
                # Create directories
                os.makedirs(file_path, exist_ok=True)
                self.logger.info(f"Created folder: {file_path}")
            else:
                # Ensure parent directories exist
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "wb") as f:
                    f.write(content)
                self.logger.debug(f"Saved file: {file_path}")

        self.logger.info("File extraction to folder completed successfully.")
        return folder
