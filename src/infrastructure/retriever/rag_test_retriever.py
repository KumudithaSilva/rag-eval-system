from typing import Any, List

from dto.test_rag_retriever import TestQuestion
from interfaces.infra.i_retriever import IRetriver
from interfaces.infra.i_vector_store import IVectorStoreService
from logs.logger_singleton import Logger


class RagRetrivever(IRetriver):
    """
    Retriever for fetching relevant context documents.

    Attributes:
        _service (IVectorStoreService): Service to create and load vector stores.
        _vectorstore (Optional[Chroma]): Cached loaded vector store instance.
        _retriever (Optional[Retriever]): Cached retriever instance created from the vector store.
    """

    def __init__(self, vectorstore: IVectorStoreService, logger=None):
        """
        Initialize the rag retriever with a vector store service.

        Args:
            vectorstore_service (IVectorStoreService): Service to create and load vector stores.
            logger (Logger, optional): A logger instance. If None, a default
                logger is created using the class name.
        """
        self._service = vectorstore
        self._vectorstore = None
        self._retriever = None
        self.logger = logger or Logger(self.__class__.__name__)

    def _ensure_loaded(self):
        """
        Load the vector store and create a retriever if not already done.
        """
        if self._vectorstore is None:
            self._vectorstore = self._service.load()
            self._retriever = self._vectorstore.as_retriever()

    def fetch_context(self, question: TestQuestion) -> List:
        """
        Retrieve relevant context documents for a question.

        Args:
            question (TestQuestion): Test question

        Return:
            List: Retrieved relevant context documents.
        """
        self._ensure_loaded()
        question_text = getattr(question, "question")

        relevent_documents = self._retriever.invoke(question_text, k=3)
        self.logger.debug(f"Retrieve relevant context documents: {relevent_documents}")

        return relevent_documents

    def fetch_contexts(self, questions: List[TestQuestion]) -> List[List[Any]]:
        """
        Retrieve relevant context documents for a questions.

        Args:
            questions (List[TestQuestion]): List of Test questions

        Return:
            List[List[Any]]: Retrieved context documents per question.
            Ex: [[doc1, doc2], [doc3, doc4]]

        """
        self._ensure_loaded()
        results = []
        for q in questions:
            question_text = getattr(q, "question")
            docs = self._retriever.invoke(question_text, k=3)
            results.append(docs)
        self.logger.debug(f"Retrieve relevant context documents: {results}")
        return results


if __name__ == "__main__":
    from infrastructure.db.chroma_vector_store import VectorStoreService
    from infrastructure.embedding.hugging_face_embeddings import (
        HuggingFaceEmbeddingModel,
    )
    from utils.rag_tests import load_tests

    test = load_tests()
    rag_test = test[:2]

    emb_model = HuggingFaceEmbeddingModel(model_name="all-MiniLM-L6-v2")
    vectorstore = VectorStoreService(embedding_model=emb_model)

    retriever = RagRetrivever(vectorstore=vectorstore)
    retriever_result = retriever.fetch_contexts(questions=rag_test)

    print(retriever_result)
