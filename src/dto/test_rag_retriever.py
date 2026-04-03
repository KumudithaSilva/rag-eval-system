from typing import Annotated
from pydantic import BaseModel, Field


class TestQuestion(BaseModel):
    question: Annotated[str, Field(description="The question to ask the RAG system")]
    keywords: Annotated[
        list[str], Field(description="Keywords that must appear in retrieved context")
    ]
    reference_answer: Annotated[
        str, Field(description="The reference answer for this question")
    ]
    category: Annotated[
        str,
        Field(description="Question category (e.g., direct_fact, spanning, temporal)"),
    ]
