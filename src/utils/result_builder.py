from dto.llm_schemas import Chunk, Result


class ResultBuilder:
    """
    Utility class to convert Chunk objects into Result objects.
    """

    @staticmethod
    def from_chunk(chunk: Chunk) -> Result:
        """
        Convert Chunk objects into Result object.

        Args:
            chunks (Chunk): Chunk objects

        Returns:
            Result: Result objects
        """
        page_content = "\n\n".join([chunk.headline, chunk.summary, chunk.original_text])
        return Result(page_content=page_content)
