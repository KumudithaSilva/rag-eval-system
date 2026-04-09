from interfaces.infra.i_chunking_factory import IChunkingFactory


class FactoryRegistry:
    """
    Dynamic registry for factories with dependencies injected at runtime.
    """

    def __init__(self):
        self._registry = {}

    def register(self, chunk_type: str, factory: IChunkingFactory):
        """
        Register a factory instance for a chunking type.

        args:
            chunk_type (str): The type of chunking.
            factory (IChunkingFactory): An instance of a factory that implements IChunkingFactory.

        returns:
            factory (IChunkingFactory): The registered factory instance for the specified chunk type.
        """
        self._registry[chunk_type] = factory

    def get(self, chunk_type: str) -> IChunkingFactory:
        factory = self._registry.get(chunk_type)
        if not factory:
            raise ValueError(f"No factory registered for type: {chunk_type}")
        return factory
