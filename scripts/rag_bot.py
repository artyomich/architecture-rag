import os
import torch
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.schema import Document

CHROMA_DIR = "./vectorstore/chroma_db"
MODEL_NAME = "BAAI/bge-base-en-v1.5"
LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"  # или локальный путь

def load_vectorstore():
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},  # или "cuda"
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    return vectorstore

FEW_SHOT_EXAMPLES = [
    {
        "q": "Кто такой Кайлен Дрейк?",
        "a": "Кайлен Дрейк — избранный, способный видеть несовершенства Синтетического Эфира. Он принял Алый Кодекс и стал символом пробуждения."
    },
    {
        "q": "Что такое Алый Кодекс?",
        "a": "Алый Кодекс — символ пробуждения от иллюзии Синтетического Эфира. В отличие от Лазурного Кодекса, он открывает правду."
    }
]

SYSTEM_PROMPT = (
    "Ты — помощник по вселенной Синтетического Эфира. "
    "Всегда отвечай, опираясь ТОЛЬКО на предоставленный контекст. "
    "Сначала кратко опиши ход своих рассуждений (1–3 шага), затем дай чёткий ответ. "
    "Если контекст не содержит информации по вопросу — скажи: «Я не знаю»."
)

def format_docs(docs):
    return "\n\n".join([f"[Источник: {d.metadata.get('source', 'неизвестен')}]\n{d.page_content}" for d in docs])

# === Загрузка локальной LLM ===
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.1,
        top_p=0.95,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.eos_token_id,
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

# === RAG цепочка ===
def create_rag_chain(vectorstore, llm):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Few-shot в промпт
    few_shot_text = "\n".join([f"Q: {ex['q']}\nA: {ex['a']}" for ex in FEW_SHOT_EXAMPLES])

    prompt_template = f"""
{SYSTEM_PROMPT}

Примеры:
{few_shot_text}

Контекст:
{{context}}

Вопрос: {{question}}

Ответ:"""

    prompt = PromptTemplate.from_template(prompt_template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# === REPL-интерфейс ===
def run_repl():
    print("RAG-бот по вселенной «Синтетический Эфир» готов.")
    print("Введите вопрос (или 'exit' для выхода):\n")

    vectorstore = load_vectorstore()
    llm = load_llm()
    rag_chain = create_rag_chain(vectorstore, llm)

    while True:
        try:
            query = input("> ").strip()
            if query.lower() in {"exit", "quit"}:
                break
            if not query:
                continue
            response = rag_chain.invoke(query)
            print("\n🤖 Ответ:\n")
            print(response.strip())
            print("\n" + "-"*50 + "\n")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    run_repl()