"""Global pytest fixtures and configuration."""

import pytest
import os
import tempfile
from unittest.mock import MagicMock, patch


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_document():
    """Create a mock LangChain Document."""
    doc = MagicMock()
    doc.page_content = "Test content"
    doc.metadata = {"source": "test.txt"}
    return doc


@pytest.fixture
def mock_documents():
    """Create mock LangChain Documents."""
    docs = []
    for i in range(3):
        doc = MagicMock()
        doc.page_content = f"Document {i} content about the Matrix universe."
        doc.metadata = {"source": f"test_{i}.txt"}
        docs.append(doc)
    return docs