from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.agents import initialize_agent, Tool
from langchain.chains import RetrievalQA

# 1️⃣ 建立文件問答系統（RAG） → 我們稍後會把它包裝成一個 Agent 可用的工具 Tool

# 讀取本地參考文件
loader = TextLoader("reference.txt", encoding="utf-8")
documents = loader.load()

# 切割成可向量化的小段落
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# 產生文件向量
embedding = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(docs, embedding)

# 將向量資料庫轉為檢索器
retriever = vectorstore.as_retriever(
    search_type="mmr",  # 多樣性優先檢索
    search_kwargs={"k": 3}
)

# 建立 RAG 問答鏈（給定問題 → 檢索 → 丟給 LLM 回答）
rag_qa = RetrievalQA.from_chain_type(
    llm=OllamaLLM(model="gemma3"),
    retriever=retriever,
    return_source_documents=True
)

# 2️⃣ 建立自定義工具 Tool（例如一個簡單的計算機）
# 👉 Tool 只是一個帶說明的函式，Agent 能根據描述決定是否使用它

def simple_calculator(query: str) -> str:
    try:
        result = eval(query)
        return f"{query} 的結果是：{result}"
    except Exception as e:
        return f"無法計算：{str(e)}"

calc_tool = Tool(
    name="Calculator",
    func=simple_calculator,
    description="使用這個工具來解決數學運算問題，例如 '3 + 5 * (2 - 1)'"
)

# 3️⃣ 將 RAG 問答系統也包裝成 Tool！
# ✅ 這就是本範例的重點：讓語意檢索 + 文件問答能力，變成 Agent 可調用的工具

rag_tool = Tool(
    name="RAG_DocQA",
    func=lambda q: rag_qa.invoke({"query": q})["result"],
    description="使用這個工具來查詢 reference.txt 文件中的知識，例如 'LangChain 的六大模組包括哪些？'"
)

# 4️⃣ 建立 Agent 並引入兩個工具：計算機 + 文件問答系統
# 🧠 Agent 會根據問題內容自動決定使用哪個 Tool（Zero-shot 推理）

agent_executor = initialize_agent(
    tools=[calc_tool, rag_tool],
    llm=OllamaLLM(model="gemma3"),
    agent_type="zero-shot-react-description",
    verbose=True
)

# 5️⃣ 提問 → 看看 Agent 是否能正確使用兩個 Tool
questions = [
    "LangChain 的六大模組包括哪些？",     # ➜ 使用 RAG Tool 查文件
    "請幫我計算 (15 + 32) * 2 是多少？",  # ➜ 使用 Calculator Tool
    "使用 LangChain 可以做哪些應用？",     # ➜ 使用 RAG Tool 查文件
    "100 除以 4 再加上 6 等於多少？"       # ➜ 使用 Calculator Tool
]

for q in questions:
    print(f"\n🟦 問題：{q}")
    print("✅ 回答：", agent_executor.invoke(q))
