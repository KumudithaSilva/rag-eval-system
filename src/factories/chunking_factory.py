from interfaces.chunking.i_chunking_strategy import IChunkingStrategy
from registry.chunking_registry import FactoryRegistry


class ChunkingFactory:
    """Dynamic chunking strategy creation using a registry."""

    def __init__(self, registry: FactoryRegistry):
        self.registry = registry

    def create(self, chunking_data: dict, documents: list) -> IChunkingStrategy:
        chunk_type = chunking_data.get("type")
        config = chunking_data.get("config", {})

        factory = self.registry.get(chunk_type)
        return factory.create(config, documents)
