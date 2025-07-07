from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
import sqlite3
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from operator import itemgetter

# ✅ 初始化 LLM 模型
llm = OllamaLLM(model="gemma3")

# ✅ 建立摘要用 Prompt
summary_prompt = ChatPromptTemplate.from_template(
    "這是目前的對話摘要：\n\n{summary}\n\n這是新的對話內容：\n\n{new_lines}\n\n請用中文更新對話摘要："
)

# ✅ 建立對話記憶鏈
def create_chain(assistant):
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", f"你是個{assistant}，請根據對話作回應"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    history_db = SQLChatMessageHistory(
        session_id=assistant,
        connection_string='sqlite:///historyMemory.db',
        table_name="chat_memory"
    )

    memory = ConversationSummaryBufferMemory(
        llm=llm,
        prompt=summary_prompt,
        return_messages=True,
        chat_memory=history_db,
        max_token_limit=1000  # 可以依需要調整摘要上限
    )

    # ✅ 改用 load_memory_variables 來檢查是否已有摘要
    current_summary = memory.load_memory_variables({}).get("history", "")
    if not current_summary:
        messages = history_db.messages
        if messages:
            # ✅ 產生摘要（只有第一次用）
            summary_text = summarize_existing_history(messages)
            memory.moving_summary_buffer = summary_text
            print(f"🧠 從歷史資料中產生摘要如下：\n{summary_text}\n")
        else:
            print("🧠 尚無記憶紀錄，這是你們的第一次對話。\n")

    return chat_prompt, memory

def list_assistants():
    conn = sqlite3.connect("historyMemory.db")
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT DISTINCT session_id FROM chat_memory")
        assistants = [row[0] for row in cursor.fetchall()]
    except sqlite3.OperationalError:
        assistants = []  # 表格尚未建立
    conn.close()
    return assistants

# ⬇️ 自訂「繁體中文」摘要提示詞
custom_stuff_prompt = PromptTemplate.from_template(
    "你是一個善於中文總結的助理。請閱讀以下對話內容，並用繁體中文簡要總結出對話的重點：\n\n{text}\n\n繁體中文摘要："
)

def summarize_existing_history(messages):
    if not messages:
        return ""
    
    text = "\n\n".join([f"{'使用者' if m.type == 'human' else '助理'}：{m.content}" for m in messages])
    docs = [Document(page_content=text)]

    # 使用自訂 prompt 建立 summarize chain
    summarize_chain = load_summarize_chain(
        llm=llm,
        chain_type="stuff",
        prompt=custom_stuff_prompt
    )
    return summarize_chain.run(docs)

# 主程式流程
if __name__ == "__main__":
    existing = list_assistants()
    if existing:
        print("📋 目前已有記憶的助理有：")
        for name in existing:
            print(f" - {name}")
    else:
        print("📋 目前還沒有任何記憶中的助理。")

    sys_msg = input('請設定助理名稱：')
    if not sys_msg.strip():
        sys_msg = '小助理'

    chat_prompt, memory = create_chain(sys_msg)

    # ✅ 顯示該助理的記憶摘要
    print(f"\n🔁 歡迎回來，{sys_msg}！")
    summary_data = memory.load_memory_variables({}).get("history", [])
    if summary_data and isinstance(summary_data, list):
        summary_text = summary_data[0].content
        print(f"🧠 上次的摘要記憶如下：\n{summary_text}\n")
    else:
        print("🧠 尚無記憶紀錄，這是你們的第一次對話。\n")

    # ✅ 建立會話鏈（透過 LCEL 組合）
    chain = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=chat_prompt,
        verbose=True
    ) | RunnableLambda(lambda x: {"response": x['response']}) | itemgetter("response")

    print()

    while True:
        msg = input("我說：")
        if not msg.strip():
            break

        if msg.strip() == "/記憶":
            messages = memory.chat_memory.messages
            if messages:
                summary_text = summarize_existing_history(messages)
                memory.moving_summary_buffer = summary_text  # 更新記憶
                print(f"\n🧠 已更新摘要記憶內容如下：\n{summary_text}\n")
            else:
                print("\n🧠 (目前尚無摘要記憶)\n")
            continue

        if msg.strip() == "/歷史":
            print("\n🗂️ SQLite 對話歷史：")
            for message in memory.chat_memory.messages:
                role = "🧑‍🦱 使用者" if message.type == "human" else "🤖 助理"
                print(f"{role}：{message.content}")
            print()
            continue

        response = chain.invoke({"input": msg})
        print(f'{sys_msg}：{response}\n')
