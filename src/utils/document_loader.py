import glob
import os
from typing import List
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from logs.logger_singleton import Logger

logger = Logger("utils-file-extractor")


def doc_convert(path: str) -> List:
    """
    Load markdown documents from a base path and attach metadata.

    Args:
        path (str): Folder path to knowledge base.

    Returns:
        List: List of processed document objects.
    """
    documents = []
    logger.info(f"Loading documents from path: {path}")
    pathfiles = glob.glob(os.path.join(path, "*"))

    for file_path in pathfiles:

        loader = DirectoryLoader(
            path=file_path,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )

        folder_docs = loader.load()
        for doc in folder_docs:
            file_path = doc.metadata.get("source")
            subfolder_name = os.path.basename(os.path.dirname(file_path))

            logger.debug(f"Processing file from: {subfolder_name}")

            doc.metadata["doc_type"] = subfolder_name
            documents.append(doc)

    logger.info(f"Total documents loaded: {len(documents)}")
    return documents
