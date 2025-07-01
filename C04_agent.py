from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.tools import tool
from langchain_community.llms import Ollama
import math

# 🦙 使用 Ollama 模型（gemma3）
llm = Ollama(model="gemma3")

# 🔧 自定義一個工具：計算平方根
@tool
def square_root(x: str) -> str:
    """計算一個數字的平方根"""
    try:
        clean_x = x.strip().replace("'", "").replace('"', '')
        return str(math.sqrt(float(clean_x)))
    except:
        return "請提供一個數字"

# 📦 建立工具清單（也可以加上其他工具）
tools = [
    Tool.from_function(func=square_root, name="Square Root", description="計算數字的平方根")
]

# 🤖 建立 Agent，並設定成反應式（REACT）模式
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 🔍 測試 Agent 問題
response = agent.invoke("請幫我算出 144 的平方根是多少")
print("回應：", response)
