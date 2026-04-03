import json
from pathlib import Path
from typing import List

from dto.test_rag_retriever import TestQuestion
from logs.logger_singleton import Logger

logger = Logger("utils-file-rag-tests")

TEST_FILE = Path("tests.jsonl")


def load_tests(file_path: str = TEST_FILE) -> List[TestQuestion]:
    """
    Load test questions from a JSONL file.

    Args:
        file_path (str): Path to the JSONL file containing test questions.

    Returns:
        List[TestQuestion]: A list of TestQuestion instances.
    """
    logger.info("RAG tests file loading...")

    tests = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                try:
                    data = json.loads(line.strip())
                    tests.append(TestQuestion(**data))
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parse error in line {line_number}: {e}")
                except TypeError as e:
                    logger.error(f"Invalid data format in line {line_number}: {e}")
    except FileNotFoundError:
        logger.error(f"Test file not found: {file_path}")

    if tests:
        logger.info(f"RAG tests file loaded successfully. {len(tests)} tests found.")
    else:
        logger.warning("No tests loaded from the file.")

    return tests
