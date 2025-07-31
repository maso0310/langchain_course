from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.chains import RetrievalQA

# 1️⃣ 載入 PDF 檔案，這裡是客家幣相關文件 Hakka.pdf
loader = PyMuPDFLoader("Hakka.pdf")
documents = loader.load()

# 2️⃣ 切割文件為小區塊，避免 LLM 無法處理過長文本
# chunk_size：每塊字元上限；chunk_overlap：重疊範圍避免語意斷裂
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# 3️⃣ 建立向量資料庫，使用 Ollama 本地模型進行嵌入（embedding）
embedding = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(docs, embedding=embedding)

# 4️⃣ 設定本地大型語言模型 LLM，如 gemma3 / llama3 / mistral 等
llm = OllamaLLM(model="gemma3")

# 5️⃣ 建立 Retriever，用於語意相似度搜尋（MMR 可提高多樣性）
retriever = vectorstore.as_retriever(
    search_type="mmr",  # 可選 similarity 或 mmr（Maximal Marginal Relevance）
    search_kwargs={"k": 10}  # 返回前 10 筆最相關段落
)

# 6️⃣ 使用 RetrievalQA 建立問答系統
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True  # 額外回傳來源段落
)

# 7️⃣ 發問，模擬使用者輸入
query = f"請問如何登記客家幣？"
prompt = f'請以繁體中文回答下列問題：{query}'

# 8️⃣ 執行 QA 查詢流程
result = qa.invoke({"query": prompt})

# 9️⃣ 輸出結果
print("✅ 回答：", result["result"])

'''
說明：nomic-embed-text 模型限制與建議
該模型為英文優化，但具備多語言支援能力（含繁體中文）
適用於一般文本語意比對與搜尋，但中文語意理解可能略不如 OpenAI 商業模型
若要強化中文語意檢索準確度，建議改用繁體中文專用模型如 bge-small-zh（需在 Ollama 安裝支援後使用）
'''