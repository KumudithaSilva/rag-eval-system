from providers.huggingface_emb_factory import HuggingFaceEmbeddingFactory
from providers.openai_emb_factory import OpenAIEmbeddingFactory

"""
Registry for embedding models.
"""

EMBEDDING_REGISTRY = {
    "HuggingFace": HuggingFaceEmbeddingFactory(),
    "OpenAI": OpenAIEmbeddingFactory(),
}
