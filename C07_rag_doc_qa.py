from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

# 1️⃣ 讀取文件
loader = TextLoader("reference.txt", encoding="utf-8")
documents = loader.load()

# 2️⃣ 文件切割
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    add_start_index=True
)
docs = text_splitter.split_documents(documents)

# 3️⃣ 建立向量資料庫（改成 Chroma）
embedding = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(docs, embedding)

# 4️⃣ 啟動 QA 系統
llm = Ollama(model="gemma3")
retriever = vectorstore.as_retriever(
    search_type="mmr",  # 多樣性導向
    search_kwargs={"k": 3}  # 取回3個相關段落以供組合
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# 5️⃣ 發問
queries = [
    "LangChain 是什麼？",
    "LangChain 支援哪些向量資料庫？",
    "如何使用 LangChain 建立文件問答系統？",
    "LangChain 的六大模組包括哪些？",
    "哪些應用可以使用 LangChain 來實作？"
]
for query in queries:
    prompt = f"請用繁體中文回答：{query}"
    result = qa.invoke({"query": prompt})
    print(f"\n🟦 問題：{query}")
    print("✅ 回答：", result["result"])
