from langchain_community.llms import Ollama
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain.agents.agent_types import AgentType
import os

# 🧠 LLM 模型初始化
llm = Ollama(model="gemma3")

# 🧠 記憶體：記錄對話上下文
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 🛠️ Tool 1：建立筆記
def create_note(content: str) -> str:
    with open("note.txt", "w", encoding="utf-8") as f:
        f.write(content)
    return "已建立新的 note.txt 筆記檔案。"

# 🛠️ Tool 2：讀取筆記內容
def read_note(_: str = "") -> str:
    if not os.path.exists("note.txt"):
        return "找不到 note.txt 檔案，請先建立。"
    with open("note.txt", "r", encoding="utf-8") as f:
        return f.read()

# 🛠️ Tool 3：追加內容
def append_note(new_text: str) -> str:
    if not os.path.exists("note.txt"):
        return "找不到 note.txt 檔案，請先建立。"
    with open("note.txt", "a", encoding="utf-8") as f:
        f.write("\n" + new_text)
    return "已將內容追加到 note.txt。"

# 🛠️ Tool 4：根據語意修改指定段落
def edit_note(instruction: str) -> str:
    if not os.path.exists("note.txt"):
        return "找不到 note.txt 檔案，請先建立。"
    with open("note.txt", "r", encoding="utf-8") as f:
        original = f.read()

    prompt = f"""以下是目前的筆記內容：
{original}

請依據以下修改指令，僅修改必要的段落：
{instruction}

請回傳完整修改後的筆記內容（保留未修改的段落）：
"""
    new_content = llm.invoke(prompt)
    with open("note.txt", "w", encoding="utf-8") as f:
        f.write(new_content)

    return "已根據指示修改 note.txt 檔案。"

# 🔧 註冊所有工具
tools = [
    Tool(name="CreateNote", func=create_note, description="建立一個新的 note.txt 筆記檔案"),
    Tool(name="ReadNote", func=read_note, description="讀取 note.txt 筆記內容"),
    Tool(name="AppendNote", func=append_note, description="追加文字到 note.txt 的末尾"),
    Tool(name="EditNote", func=edit_note, description="根據語意指令修改 note.txt 的內容")
]

# 🤖 初始化 Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    memory=memory,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# 💬 使用範例
print(agent.invoke("幫我建立一個筆記，內容是：今天學習 LangChain 的 Tool 機制"))
print(agent.invoke("幫我加上一句：未來可以試著整合 LINE Bot 一起使用"))
print(agent.invoke("請唸出整份筆記"))
print(agent.invoke("請把剛剛那句話修改為：未來可以整合 LINE Bot 和 Notion API 做自動筆記整理"))
