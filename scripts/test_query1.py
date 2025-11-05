retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
query = "Кто такой Кайлен Дрейк и как он связан с Алым Кодексом?"
results = retriever.invoke(query)

print("\n🔍 Запрос:", query)
for i, doc in enumerate(results):
    print(f"\n📄 Чанк {i+1} (источник: {os.path.basename(doc.metadata['source'])}):")
    print(doc.page_content[:300] + "...")