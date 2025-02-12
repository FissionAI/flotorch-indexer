import unittest

from chunking.chunking import Chunk
from chunking.fixedsize_chunking import FixedSizeChunker
from chunking.hierarical_chunking import HieraricalChunker


class TestChunk(unittest.TestCase):
    def test_chunk_initialization(self):
        data = "Sample data"
        chunk = Chunk(data)
        self.assertEqual(chunk.data, data)
        self.assertIsNone(chunk.child_data)
        self.assertTrue(chunk.id)

    def test_add_child(self):
        chunk = Chunk("Parent data")
        chunk.add_child("Child data 1")
        chunk.add_child("Child data 2")
        self.assertEqual(chunk.child_data, ["Child data 1", "Child data 2"])

    def test_chunk_str(self):
        chunk = Chunk("Parent data")
        chunk.add_child("Child data 1")
        chunk.add_child("Child data 2")
        self.assertEqual(str(chunk), "Parent Chunk: Parent data, Chunk: ['Child data 1', 'Child data 2']")


class TestFixedSizeChunk(unittest.TestCase):
    def setUp(self):
        self.chunk_size = 10
        self.chunk_overlap = 20  # This should be less than chunk_size, change this value
        self.fixed_chunk = FixedSizeChunker(self.chunk_size, self.chunk_overlap)

    def test_initialization(self):
        # Assert chunking size and overlap are calculated correctly
        self.assertEqual(self.fixed_chunk.chunk_size, 40)  # 4 tokens per character * chunk_size
        self.assertEqual(self.fixed_chunk.chunk_overlap, 8)  # 20% overlap

    def test_invalid_chunk_size(self):
        with self.assertRaises(ValueError):
            FixedSizeChunker(0, 20)

    def test_invalid_chunk_overlap(self):
        # Make sure chunk_overlap is smaller than chunk_size
        with self.assertRaises(ValueError):
            FixedSizeChunker(10, 100)  # chunk_overlap exceeds chunk_size

    def test_chunk_empty_data(self):
        with self.assertRaises(ValueError):
            list(self.fixed_chunk.chunk(""))  # Empty data should raise ValueError

    def test_chunk_functionality(self):
        data = "This is a test string for chunking."
        chunks = self.fixed_chunk.chunk(data)
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertIsInstance(chunk, Chunk)


class TestHieraricalChunk(unittest.TestCase):
    def setUp(self):
        self.chunk_size = 10
        self.chunk_overlap = 20  # This should be less than chunk_size, change this value
        self.parent_chunk_size = 50
        self.hierarchical_chunk = HieraricalChunker(self.chunk_size, self.chunk_overlap, self.parent_chunk_size)

    def test_initialization(self):
        # Assert chunking size and overlap are calculated correctly
        self.assertEqual(self.hierarchical_chunk.chunk_size, 40)  # 4 tokens per character * chunk_size
        self.assertEqual(self.hierarchical_chunk.parent_chunk_size, 200)  # 4 tokens per character * parent_chunk_size

    def test_invalid_parent_chunk_size(self):
        with self.assertRaises(ValueError):
            HieraricalChunker(10, 20, 0)

        with self.assertRaises(ValueError):
            HieraricalChunker(10, 20, 5)  # Parent chunking size must be larger than child chunking size

    def test_chunk_empty_data(self):
        with self.assertRaises(ValueError):
            list(self.hierarchical_chunk.chunk(""))  # Empty data should raise ValueError

    def test_chunk_functionality(self):
        data = "This is a test string for hierarchical chunking. It should split into parent and child chunks."
        chunks = self.hierarchical_chunk.chunk(data)
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertIsInstance(chunk, Chunk)
            self.assertIsNotNone(chunk.child_data)
            self.assertGreater(len(chunk.child_data), 0)

    def test_parent_child_relation(self):
        data = "This is a test string for hierarchical chunking. It should split into parent and child chunks."
        chunks = self.hierarchical_chunk.chunk(data)
        parent_chunk = chunks[0]
        self.assertIsInstance(parent_chunk, Chunk)
        self.assertIsInstance(parent_chunk.child_data, list)
        self.assertGreater(len(parent_chunk.child_data), 0)


if __name__ == "__main__":
    unittest.main()
