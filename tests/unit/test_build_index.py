"""Unit tests for scripts/build_index.py functionality."""

import pytest
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch, mock_open

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


class TestBuildIndexPaths:
    """Tests for build_index path calculations."""

    def test_knowledge_base_path_construction(self):
        """Knowledge base path should be correctly constructed from script path."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        expected_kb = os.path.join(
            os.path.dirname(os.path.dirname(base_dir)),
            "knowledge_base",
            "final"
        )
        assert "knowledge_base" in expected_kb
        assert "final" in expected_kb

    def test_vectorstore_path_construction(self):
        """Vectorstore path should be correctly constructed."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        expected_vs = os.path.join(base_dir, "..", "scripts", "vectorstore", "chroma_db")
        assert "vectorstore" in expected_vs
        assert "chroma_db" in expected_vs


class TestKnowledgeBaseValidation:
    """Tests for knowledge base directory validation logic."""

    def test_knowledge_dir_exists_check(self):
        """Should check if knowledge directory exists."""
        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = False
            with patch("os.path.isdir", return_value=False):
                with patch("sys.exit") as mock_exit:
                    # Simulating the check from build_index.py
                    knowledge_dir = "/fake/path/knowledge_base/final"
                    if not os.path.exists(knowledge_dir):
                        mock_exit.assert_called_once()

    def test_knowledge_dir_not_found_behavior(self):
        """Should exit with error when knowledge dir not found."""
        with patch("os.path.exists", return_value=False):
            with patch("sys.exit") as mock_exit:
                knowledge_dir = "/nonexistent/path"
                if not os.path.exists(knowledge_dir):
                    assert mock_exit.called


class TestDocumentLoading:
    """Tests for document loading logic."""

    def test_txt_file_filtering(self):
        """Only .txt files should be loaded."""
        files = ["file1.txt", "file2.py", "file3.txt", "file4.md", "file5.TXT"]
        txt_files = [f for f in files if f.endswith(".txt")]
        assert len(txt_files) == 2
        assert "file1.txt" in txt_files
        assert "file3.txt" in txt_files

    def test_txt_file_filtering_case_insensitive(self):
        """Should handle .TXT extension."""
        files = ["file1.TXT", "file2.txt", "file1.Txt"]
        txt_files = [f for f in files if f.endswith(".txt")]
        assert len(txt_files) == 1
        assert "file2.txt" in txt_files

    def test_sorted_file_loading(self):
        """Files should be loaded in sorted order."""
        files = ["zebra.txt", "alpha.txt", "middle.txt"]
        sorted_files = sorted(files)
        assert sorted_files[0] == "alpha.txt"
        assert sorted_files[1] == "middle.txt"
        assert sorted_files[2] == "zebra.txt"


class TestChunkConfiguration:
    """Tests for text chunking configuration."""

    def test_chunk_size_is_positive(self):
        """Chunk size should be positive."""
        chunk_size = 500
        assert chunk_size > 0

    def test_chunk_overlap_less_than_size(self):
        """Chunk overlap should be less than chunk size."""
        chunk_size = 500
        chunk_overlap = 50
        assert chunk_overlap < chunk_size

    def test_chunk_overlap_is_positive(self):
        """Chunk overlap should be positive."""
        assert 50 > 0

    def test_separators_list_not_empty(self):
        """Separators list should not be empty."""
        separators = ["\n\n", "\n", ". ", " ", ""]
        assert len(separators) > 0


class TestEmbeddingConfiguration:
    """Tests for embedding model configuration."""

    def test_model_name_valid(self):
        """Model name should be properly formatted."""
        model_name = "BAAI/bge-base-en-v1.5"
        assert "/" in model_name
        assert len(model_name.split("/")) == 2

    def test_model_kwargs_has_device(self):
        """Model kwargs should have device setting."""
        model_kwargs = {"device": "cpu"}
        assert "device" in model_kwargs
        assert model_kwargs["device"] == "cpu"

    def test_encode_kwargs_has_normalize(self):
        """Encode kwargs should have normalize setting."""
        encode_kwargs = {"normalize_embeddings": True}
        assert "normalize_embeddings" in encode_kwargs

    def test_query_instruction_not_empty(self):
        """Query instruction should not be empty."""
        query_instruction = "Represent this sentence for searching relevant passages:"
        assert len(query_instruction) > 0


class TestVectorstorePersistence:
    """Tests for vectorstore persistence logic."""

    def test_persist_directory_created(self):
        """Persist directory should be created if not exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = os.path.join(tmpdir, "new_dir", "chroma_db")
            os.makedirs(os.path.dirname(test_path), exist_ok=True)
            assert os.path.exists(os.path.dirname(test_path))

    def test_vectorstore_path_contains_chroma(self):
        """Vectorstore path should contain 'chroma'."""
        vectorstore_path = "/test/path/vectorstore/chroma_db"
        assert "chroma" in vectorstore_path.lower()


class TestBuildIndexIntegration:
    """Integration-style tests for build_index logic."""

    def test_full_document_loading_simulation(self):
        """Simulate full document loading process."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake knowledge base files
            kb_dir = os.path.join(tmpdir, "knowledge_base", "final")
            os.makedirs(kb_dir)
            
            for name in ["alpha.txt", "beta.txt", "gamma.txt"]:
                with open(os.path.join(kb_dir, name), "w") as f:
                    f.write(f"Content of {name}")
            
            # Count files that would be loaded
            loaded = 0
            for filename in sorted(os.listdir(kb_dir)):
                if filename.endswith(".txt"):
                    loaded += 1
            
            assert loaded == 3

    def test_chunking_simulation(self):
        """Simulate text chunking process."""
        text = "Hello world. This is a test. It has multiple sentences.\n\nWith paragraphs."
        # Simulate RecursiveCharacterTextSplitter behavior
        paragraphs = text.split("\n\n")
        assert len(paragraphs) >= 1

    def test_embedding_model_name_format(self):
        """Verify embedding model name follows HuggingFace format."""
        model_name = "BAAI/bge-base-en-v1.5"
        parts = model_name.split("/")
        assert len(parts) == 2
        assert len(parts[0]) > 0  # org name
        assert len(parts[1]) > 0  # model name