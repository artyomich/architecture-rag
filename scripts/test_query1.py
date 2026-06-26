#!/usr/bin/env python3
"""Тестовый скрипт для проверки работы RAG-бота."""

import os
import sys

# Добавляем путь к скриптам
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# === Загрузка векторного хранилища ===
CHROMA_DIR = "./vectorstore/chroma_db"
MODEL_NAME = "BAAI/bge-base-en-v1.5"

embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
)

# === Тестовые запросы ===
test_queries = [
    "Кто такой Кайлен Дрейк и как он связан с Алым Кодексом?",
    "Что такое Синтетический Эфир?",
    "Какая столица у планеты Ти'лора?",
]

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print("=" * 60)
print("🧪 Тестирование RAG-бота")
print("=" * 60)

for query in test_queries:
    print(f"\n🔍 Запрос: {query}")
    print("-" * 40)
    results = retriever.invoke(query)
    for i, doc in enumerate(results):
        source = os.path.basename(doc.metadata.get("source", "неизвестен"))
        print(f"\n📄 Чанк {i + 1} (источник: {source}):")
        print(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
    print()

print("=" * 60)
print("✅ Тестирование завершено")
print("=" * 60)