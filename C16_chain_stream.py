from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# ✅ 初始化支援串流的模型
llm = ChatOllama(model="gemma3", stream=True)

# ✅ 建立對話提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位親切的中文 AI 助理"),
    ("human", "{input}")
])

# ✅ 串接提示與模型
chain = prompt | llm

# ✅ 執行串流輸出
for chunk in chain.stream({"input": "請幫我簡單介紹一下台灣"}):
    print(chunk.content, end="", flush=True)
