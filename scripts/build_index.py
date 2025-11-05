import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma

# === Загрузка документов ===
docs = []
knowledge_dir = "knowledge_base/final"

for filename in os.listdir(knowledge_dir):
    if filename.endswith(".txt"):
        filepath = os.path.join(knowledge_dir, filename)
        loader = TextLoader(filepath, encoding="utf-8")
        docs.extend(loader.load())

print(f"✅ Загружено {len(docs)} документов")

# === Разбиение на чанки ===
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # ~120–150 слов
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)
splits = text_splitter.split_documents(docs)
print(f"✂️ Разбито на {len(splits)} чанков")

model_name = "BAAI/bge-base-en-v1.5"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}

embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    query_instruction="Represent this sentence for searching relevant passages:"
)

vectorstore_path = "./vectorstore/chroma_db"

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=vectorstore_path
)

print(f"💾 Векторный индекс сохранён в: {vectorstore_path}")