from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_ollama import ChatOllama

# 模擬簡單的記憶儲存系統（以 dict 暫存）
memory_store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in memory_store:
        memory_store[session_id] = ChatMessageHistory()
    return memory_store[session_id]

# 建立 Prompt 模板（含歷史訊息插槽）
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一個親切的 AI 助理，請用繁體中文回答。"),
    MessagesPlaceholder(variable_name="history"),  # 插入歷史訊息
    ("human", "{input}"),
])

# 模型與輸出解析器
llm = ChatOllama(model="gemma3")
parser = StrOutputParser()

# 建立可串接記憶的對話 chain
chain = prompt | llm | parser

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,                # session_id ➜ history 的函式
    input_messages_key="input",         # 使用者輸入對應的變數名
    history_messages_key="history",     # 對應 prompt 中的 placeholder 名稱
)

# 測試對話流程
session_id = "user_001"
while True:
    user_input = input("你：")
    response = chain_with_history.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )
    print("AI：", response)
