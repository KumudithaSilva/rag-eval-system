import os
import glob
import json
from typing import List
import logging

from dto.test_rag_retriever import TestQuestion

logger = logging.getLogger(__name__)


def load_tests(folder_path: str) -> List[TestQuestion]:
    """
    Load test questions from the JSONL file inside a folder.

    Args:
        folder_path (str): Path to the folder containing one JSONL file.

    Returns:
        List[TestQuestion]: A list of TestQuestion instances.
    """
    logger.info(f"Looking for JSONL file inside folder: {folder_path}")
    tests = []

    # Find JSONL files in the folder
    files = glob.glob(os.path.join(folder_path, "*.jsonl"))

    if not files:
        logger.error(f"No JSONL file found in folder: {folder_path}")
        return tests

    if len(files) > 1:
        logger.warning(
            f"More than one JSONL file found, using the first one: {files[0]}"
        )

    file_path = files[0]
    logger.info(f"Loading JSONL file: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                try:
                    data = json.loads(line.strip())
                    tests.append(TestQuestion(**data))
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parse error in line {line_number}: {e}")
                except TypeError as e:
                    logger.error(f"Type error in line {line_number}: {e}")
    except Exception as e:
        logger.error(f"Error opening test file {file_path}: {e}")

    return tests
