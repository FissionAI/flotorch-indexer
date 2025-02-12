from abc import ABC, abstractmethod
from typing import List, Dict

from chunking.chunking import Chunk

"""
This class is responsible for embedding the text using the Llama model.
"""
class EmbeddingMetadata:
    """
    Initializes the EmbeddingMetadata class.
    :param input_tokens: The number of input tokens.
    :param latency_ms: The latency in milliseconds.
    """
    def __init__(self, input_tokens: int, latency_ms: int):
        self.input_tokens = input_tokens
        self.latency_ms = latency_ms
    
    def append(self, metadata: 'EmbeddingMetadata'):
        self.input_tokens += int(metadata.input_tokens)
        self.latency_ms += int(metadata.latency_ms)


class Embeddings:
    """
    Initializes the Embeddings class.
    :param embeddings: The embeddings.
    :param metadata: The metadata.
    """
    def __init__(self, embeddings: List[List[float]], metadata: EmbeddingMetadata):
        self.embeddings = embeddings
        self.metadata = metadata

    def to_json(self) -> Dict:
        {
            "embeddings": self.embeddings,
            "metadata": {
                    "input_tokens": self.metadata.input_tokens,
                    "latency_ms": self.metadata.latency_ms
                }
        }

class EmbeddingList:
    def __init__(self):
        self.embeddings: List[Embeddings] = []
        self.metadata = EmbeddingMetadata(0, 0)

    def append(self, embeddings: Embeddings):
        self.embeddings.append(embeddings)
        self.metadata.append(embeddings.metadata)

"""
This class is responsible for embedding the text."""
class BaseEmbedding(ABC):
    """
    Initializes the BaseEmbedding class.
    :param dimensions: The dimensions of the embedding.
    :param normalize: Normalize the embedding.
    """

    def __init__(self,  model_id: str, region: str, dimensions: int = 256, normalize: bool = True) -> None:
        super().__init__()
        self.model_id = model_id
        self.region = region
        self.dimension = dimensions
        self.normalize = normalize

    """
    Prepares the chunk for embedding.
    :param chunk: The chunk to be embedded.
    :return: The prepared chunk.
    """
    @abstractmethod
    def _prepare_chunk(self, chunk: Chunk) -> Dict:
        pass

    """
    Embeds the chunk.
    :param chunk: The chunk to be embedded.
    :return: The embeddings.
    """
    @abstractmethod
    def embed(self, chunk: Chunk) -> Embeddings:
        pass

    """
    Embeds the list of chunks.
    :param chunks: The list of chunks to be embedded.
    :return: The list of embeddings.
    """
    def embed_list(self, chunks: List[Chunk]) -> EmbeddingList:
        embedding_list = EmbeddingList()
        if not isinstance(chunks, list):
            return embedding_list.append(self.embed(chunks))
        for chunk in chunks:
            embedding = self.embed(chunk)
            embedding_list.append(embedding)
        return embedding_list