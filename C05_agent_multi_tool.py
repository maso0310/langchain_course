from langchain.agents import Tool, initialize_agent, AgentType
from langchain_community.llms import Ollama
from langchain.tools import tool
import math

# 🧠 模型（請根據你本地支援的 Ollama 模型改名）
llm = Ollama(model="gemma3")

# 🛠 自訂工具 1：平方根計算器
@tool
def square_root(x: str) -> str:
    """計算輸入數字的平方根"""
    try:
        clean_x = x.strip().replace('"', '').replace("'", '')
        return str(math.sqrt(float(clean_x)))
    except:
        return "請提供一個正確的數字"

# 🛠 自訂工具 2：假裝搜尋工具（這裡先模擬，日後可整合 Google/Bing API）
@tool
def fake_search(query: str) -> str:
    """模擬搜尋並回傳假資料"""
    if "蘋果創辦人" in query:
        return "蘋果的創辦人是 Steve Jobs。"
    return f"你查詢的是：{query}（這是模擬搜尋結果）"

# 🔗 工具列表
tools = [
    Tool.from_function(func=square_root, name="Square Root", description="計算數字的平方根"),
    Tool.from_function(func=fake_search, name="Search", description="搜尋常識問題")
]

# 🤖 建立 Agent（使用 ReAct 模式）
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 🔍 測試
response = agent.invoke("請問144的平方根是多少？")
print("平方根結果：", response)
response = agent.invoke("蘋果創辦人是誰？")
print("搜尋結果：", response)
