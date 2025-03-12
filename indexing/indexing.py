from flotorch_core.chunking.chunking import BaseChunker
from flotorch_core.embedding.embedding import BaseEmbedding, EmbeddingList
from flotorch_core.reader.pdf_reader import PDFReader


class Index:
    def __init__(self, pdf_reader: PDFReader, chunker: BaseChunker, embedder: BaseEmbedding):
        self.pdf_reader = pdf_reader
        self.chunker = chunker
        self.embedder = embedder

    def index(self, path: str) -> EmbeddingList:
        text = self.pdf_reader.read_pdf(path)
        chunks = self.chunker.chunk_list(text)
        embeddings = self.embedder.embed_list(chunks)
        return embeddings