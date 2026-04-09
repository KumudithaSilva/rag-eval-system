from interfaces.embedding.i_embedding import IEmbeddingModel


class EmbeddingFactoryRegistry:
    """
    Dynamic registry for factories with dependencies injected at runtime.
    """

    def __init__(self):
        self._registry = {}

    def register(self, emb_type: str, factory: IEmbeddingModel):
        """
        Register a factory instance for embedding type.

        args:
            emb_type (str): The type of embedding.
            factory (IEmbeddingModel): The factory instance to register.

        returns:
            factory (IEmbeddingModel): The registered factory instance.
        """
        self._registry[emb_type] = factory

    def get(self, emb_type: str) -> IEmbeddingModel:
        factory = self._registry.get(emb_type)
        if not factory:
            raise ValueError(f"No factory registered for embedding type: {emb_type}")
        return factory
