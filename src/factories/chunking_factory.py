from interfaces.chunking.i_chunking_strategy import IChunkingStrategy
from registry.chunking_registry import CHUNKING_REGISTRY


class ChunkingFactory:
    """Create chunking strategy instances."""

    @staticmethod
    def create(chunking_data: dict, path: str) -> IChunkingStrategy:
        chunk_type = chunking_data.get("type")
        config = chunking_data.get("config", {})

        factory = CHUNKING_REGISTRY.get(chunk_type)

        if not factory:
            raise ValueError(f"Unsupported chunking type: {chunk_type}")

        return factory.create(config, path)
