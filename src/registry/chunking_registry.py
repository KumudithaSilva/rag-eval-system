from providers.default_chunking_factory import DefaultChunkingFactory
from providers.llm_chunking_factory import LLMChunkingFactory

"""
Registry for chunking strategies.
"""

CHUNKING_REGISTRY = {
    "Default Chunking": DefaultChunkingFactory(),
    "LLM Chunking": LLMChunkingFactory(),
}
