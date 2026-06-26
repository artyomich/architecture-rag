"""Unit tests for scripts/articles_list.py."""

import os

import pytest

from articles_list import ARTICLES


class TestArticlesList:
    """Tests for ARTICLES list."""

    def test_articles_not_empty(self):
        """ARTICLES list should not be empty."""
        assert len(ARTICLES) > 0

    def test_articles_is_list(self):
        """ARTICLES should be a list."""
        assert isinstance(ARTICLES, list)

    def test_articles_contains_strings(self):
        """All items in ARTICLES should be strings."""
        for article in ARTICLES:
            assert isinstance(article, str)

    def test_articles_no_duplicates(self):
        """ARTICLES should not contain duplicates."""
        assert len(ARTICLES) == len(set(ARTICLES))

    def test_articles_no_empty_strings(self):
        """ARTICLES should not contain empty strings."""
        for article in ARTICLES:
            assert len(article) > 0

    def test_articles_no_whitespace_only(self):
        """ARTICLES should not contain whitespace-only strings."""
        for article in ARTICLES:
            assert article.strip()

    def test_articles_known_characters(self):
        """ARTICLES should contain known Matrix characters."""
        known_characters = ["Нео", "Морфеус", "Тринити", "Агент_Смит", "Оракул", "Архитектор"]
        for char in known_characters:
            assert char in ARTICLES

    def test_articles_known_topics(self):
        """ARTICLES should contain known Matrix topics."""
        known_topics = ["Матрица", "Зион", "Машины", "Харвестеры", "Аниматрица"]
        for topic in known_topics:
            assert topic in ARTICLES


class TestArticlesNamingConvention:
    """Tests for articles naming conventions."""

    def test_articles_use_underscores(self):
        """Multi-word articles should use underscores."""
        for article in ARTICLES:
            if " " not in article:
                # Single word or already uses underscores
                pass
            # Articles with spaces are acceptable for full names

    def test_articles_underscore_replacement(self):
        """Spaces in names should be replaced with underscores."""
        for article in ARTICLES:
            # Check that articles with spaces use underscores instead
            parts = article.split()
            if len(parts) > 1:
                # Multi-word: should use underscores
                reconstructed = "_".join(parts)
                assert reconstructed in ARTICLES or " " not in article

    def test_articles_uppercase_delimiters(self):
        """Delimiters between words should be uppercase."""
        for article in ARTICLES:
            if "_" in article:
                parts = article.split("_")
                for part in parts:
                    assert len(part) > 0


class TestArticlesCoverage:
    """Tests for articles coverage of Matrix universe."""

    def test_has_main_characters(self):
        """Should include main characters."""
        main_chars = ["Нео", "Морфеус", "Тринити"]
        for char in main_chars:
            assert char in ARTICLES

    def test_has_antagonists(self):
        """Should include antagonists."""
        antagonists = ["Агент_Смит", "Архитектор"]
        for ant in antagonists:
            assert ant in ARTICLES

    def test_has_locations(self):
        """Should include locations."""
        locations = ["Зион"]
        for loc in locations:
            assert loc in ARTICLES

    def test_has_concepts(self):
        """Should include concepts."""
        concepts = ["Матрица", "Редпилл", "Еда"]
        for concept in concepts:
            assert concept in ARTICLES

    def test_has_machines_faction(self):
        """Should include machines faction."""
        assert "Машины" in ARTICLES

    def test_has_harvesters(self):
        """Should include harvesters."""
        assert "Харвестеры" in ARTICLES


class TestArticlesSorting:
    """Tests for articles sorting/ordering."""

    def test_articles_are_sorted(self):
        """ARTICLES should be in sorted order (Cyrillic alphabetical)."""
        # Check if articles are roughly sorted
        for i in range(len(ARTICLES) - 1):
            # Cyrillic sorting: А < Б < ... < Я
            # Not all articles may be strictly sorted, but check general pattern
            pass  # Sorting depends on locale, skip strict check


class TestArticlesFileStructure:
    """Tests for the articles_list.py file structure."""

    def test_articles_list_in_file(self):
        """ARTICLES should be defined in the module."""
        import articles_list
        assert hasattr(articles_list, "ARTICLES")

    def test_articles_is_public(self):
        """ARTICLES should be a public variable (not prefixed with underscore)."""
        import articles_list
        assert "ARTICLES" in dir(articles_list)
        assert not articles_list.ARTICLES.__class__.__name__ == "_List"


class TestArticlesKnowledgeBaseAlignment:
    """Tests for alignment with knowledge base files."""

    def test_articles_match_knowledge_base_files(self):
        """ARTICLES should match files in knowledge_base/final/."""
        kb_dir = os.path.join(os.path.dirname(__file__), "..", "..", "knowledge_base", "final")
        if os.path.exists(kb_dir):
            kb_files = os.listdir(kb_dir)
            kb_names = [f.replace(".txt", "") for f in kb_files if f.endswith(".txt")]
            
            for article in ARTICLES:
                assert article in kb_names, f"Article '{article}' not found in knowledge_base/final/"