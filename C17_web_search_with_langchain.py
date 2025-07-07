from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.agent import AgentOutputParser
from langchain.agents.agent_types import AgentType
from langchain.agents.format_scratchpad import format_to_openai_functions

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.llms import Ollama  # 或使用 OpenAI
import os

# ✅ 初始化 LLM（這裡用本地 Ollama 模型）
llm = Ollama(model="gemma3")

# ✅ 初始化 DuckDuckGo 搜尋工具
search = DuckDuckGoSearchRun()

# ✅ 將搜尋包裝成 Tool
tools = [
    Tool(
        name="duckduckgo_search",
        func=search.run,
        description="當你需要從網路搜尋最新資訊時可以使用這個工具，整理出繁體中文結論"
    )
]

# ✅ 建立 Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True
)

# ✅ 測試提問
question = "台灣 2025 年金曲獎獲獎人？請將所有的獎項與獲獎人整理成表格"
response = agent.run(question)
print(f"\n🤖 AI 回應：{response}")
