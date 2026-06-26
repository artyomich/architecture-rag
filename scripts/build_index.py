#!/usr/bin/env python3
"""Скрипт для построения векторного индекса из базы знаний."""

import os
import sys

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma

# === Загрузка документов ===
docs = []
knowledge_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge_base", "final")

if not os.path.exists(knowledge_dir):
    print(f"❌ Папка не найдена: {knowledge_dir}")
    sys.exit(1)

for filename in sorted(os.listdir(knowledge_dir)):
    if filename.endswith(".txt"):
        filepath = os.path.join(knowledge_dir, filename)
        loader = TextLoader(filepath, encoding="utf-8")
        docs.extend(loader.load())
        print(f"  📄 Загружен: {filename}")

print(f"\n✅ Всего загружено документов: {len(docs)}")

# === Разбиение на чанки ===
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
)
splits = text_splitter.split_documents(docs)
print(f"✂️ Разбито на {len(splits)} чанков")

# === Настройка эмбеддингов ===
model_name = "BAAI/bge-base-en-v1.5"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
query_instruction = "Represent this sentence for searching relevant passages:"

embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    query_instruction=query_instruction,
)

# === Создание векторного хранилища ===
base_dir = os.path.dirname(os.path.abspath(__file__))
vectorstore_path = os.path.join(base_dir, "vectorstore", "chroma_db")

os.makedirs(os.path.dirname(vectorstore_path), exist_ok=True)

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=vectorstore_path,
)

print(f"💾 Векторный индекс сохранён в: {vectorstore_path}")
print(f"📊 Всего векторов: {vectorstore._collection.count()}")