from typing import Dict
import zipfile
import rarfile


def extract_zip(path: str) -> Dict[str, bytes]:
    """
    Extract all files from a ZIP archive.

    Args:
        path (str): Path to the ZIP file.

    Returns:
        Dict[str, bytes]: Extracted file names and their binary content.
    """
    extracted_files: Dict[str, bytes] = {}

    with zipfile.ZipFile(path, "r") as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.is_dir():
                extracted_files[file_info.filename] = b""
            else:
                extracted_files[file_info.filename] = zip_ref.read(file_info.filename)
    return extracted_files


def extract_rar(path: str) -> Dict[str, bytes]:
    """
    Extract all files from a RAR archive.

    Args:
        path (str): Path to the RAR file.

    Returns:
        Dict[str, bytes]: Extracted file names and their binary content.
    """
    extracted_files: Dict[str, bytes] = {}

    with rarfile.RarFile(path) as rar_ref:
        for file_info in rar_ref.infolist():
            if file_info.isdir():
                extracted_files[file_info.filename] = b""
            else:
                extracted_files[file_info.filename] = rar_ref.read(file_info.filename)
    return extracted_files
