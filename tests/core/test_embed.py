import unittest

from chunking.chunking import Chunk
from embedding.cohere_embedding import CohereEmbedding
from embedding.titanv1_embedding import TitanV1Embedding
from embedding.titanv2_embedding import TitanV2Embedding


class TestCohereEmbed(unittest.TestCase):
    def setUp(self):
        self.embedder = CohereEmbedding(model_id="test_model", region="us-west-2")

    def test_prepare_chunk(self):
        chunk = Chunk(data="test data")
        payload = self.embedder._prepare_chunk(chunk)
        self.assertEqual(payload, {"texts": ["test data"], "input_type": "search_document"})


class TestTitanV1Embed(unittest.TestCase):
    def setUp(self):
        self.embedder = TitanV1Embedding(model_id="test_model", region="us-west-2")

    def test_prepare_chunk(self):
        chunk = Chunk(data="test data")
        payload = self.embedder._prepare_chunk(chunk)
        self.assertEqual(payload, {"inputText": "test data", "embeddingConfig": {"outputEmbeddingLength": 256}})

    def test_extract_embedding(self):
        response = {"embedding": [0.1, 0.2, 0.3]}
        embedding = self.embedder.extract_embedding(response)
        self.assertEqual(embedding, [0.1, 0.2, 0.3])


class TestTitanV2Embed(unittest.TestCase):
    def setUp(self):
        self.embedder = TitanV2Embedding(model_id="test_model", region="us-west-2")

    def test_prepare_chunk(self):
        chunk = Chunk(data="test data")
        payload = self.embedder._prepare_chunk(chunk)
        self.assertEqual(payload, {"inputText": "test data", "dimensions": 256, "normalize": True})

    def test_extract_embedding(self):
        response = {"embedding": [0.1, 0.2, 0.3]}
        embedding = self.embedder.extract_embedding(response)
        self.assertEqual(embedding, [0.1, 0.2, 0.3])


if __name__ == '__main__':
    unittest.main()
