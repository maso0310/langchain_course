# ✅ 載入 LangChain 所需模組
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

# ✅ 初始化本地Ollama模型
llm = OllamaLLM(model="gemma3")

# ✅ 建立「對話摘要」的提示模板
# 🔍 這段 prompt 會在每次記憶更新時使用，幫我們把舊摘要與新對話一起傳給模型，再由模型輸出新的摘要
summary_prompt = ChatPromptTemplate.from_template(
    "這是目前的對話摘要：\n\n{summary}\n\n這是新的對話內容：\n\n{new_lines}\n\n請用中文更新對話摘要："
)

# ✅ 建立「會話流程」的函式（會根據助理名稱建立記憶與提示）
def create_chain(assistant):
    # 🔍 設定聊天提示：系統角色 + 歷史對話 + 使用者輸入
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", f"你是個{assistant}，請根據對話作回應"),
        MessagesPlaceholder(variable_name="history"),  # ⬅️ 插入記憶內容
        ("human", "{input}")  # ⬅️ 最新輸入的訊息
    ])

    # 🔍 建立 SQLite 聊天記憶資料庫，依據助理名稱儲存每段對話
    history_db = SQLChatMessageHistory(
        session_id=assistant,
        connection_string='sqlite:///historyMemory.db',
        table_name="chat_memory"
    )

    # 🔍 使用 ConversationSummaryBufferMemory：將舊對話自動濃縮摘要，保留核心內容
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        prompt=summary_prompt,
        return_messages=True,
        chat_memory=history_db,
        max_token_limit=1000  # ⬅️ 可控制摘要長度（視模型支援度調整）
    )

    # 🔍 判斷是否已經有摘要記憶，若沒有且有舊對話記錄，則第一次手動建立摘要
    current_summary = memory.load_memory_variables({}).get("history", "")
    if not current_summary:
        messages = history_db.messages
        if messages:
            summary_text = summarize_existing_history(messages)
            memory.moving_summary_buffer = summary_text  # ⬅️ 填入記憶摘要
            print(f"🧠 從歷史資料中產生摘要如下：\n{summary_text}\n")
        else:
            print("🧠 尚無記憶紀錄，這是你們的第一次對話。\n")

    return chat_prompt, memory

# ✅ 顯示所有曾經建立過記憶的助理名稱
def list_assistants():
    conn = sqlite3.connect("historyMemory.db")
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT DISTINCT session_id FROM chat_memory")
        assistants = [row[0] for row in cursor.fetchall()]
    except sqlite3.OperationalError:
        assistants = []  # ⬅️ 若表格尚未建立，回傳空陣列
    conn.close()
    return assistants

# ✅ 自訂中文摘要的 Prompt 模板
custom_stuff_prompt = PromptTemplate.from_template(
    "你是一個善於中文總結的助理。請閱讀以下對話內容，並用繁體中文簡要總結出對話的重點：\n\n{text}\n\n繁體中文摘要："
)

# ✅ 將整段舊對話進行摘要（只用於第一次建立記憶）
def summarize_existing_history(messages):
    if not messages:
        return ""
    
    # 🔍 將歷史訊息合併為文字，每句開頭加上角色名稱
    text = "\n\n".join([f"{'使用者' if m.type == 'human' else '助理'}：{m.content}" for m in messages])
    docs = [Document(page_content=text)]

    # 🔍 使用 summarize chain 加上我們自定的繁體中文提示
    summarize_chain = load_summarize_chain(
        llm=llm,
        chain_type="stuff",
        prompt=custom_stuff_prompt
    )
    return summarize_chain.run(docs)

# ✅ 主程式開始（互動入口）
if __name__ == "__main__":
    # 🔍 先列出目前有哪些已儲存的助理
    existing = list_assistants()
    if existing:
        print("📋 目前已有記憶的助理有：")
        for name in existing:
            print(f" - {name}")
    else:
        print("📋 目前還沒有任何記憶中的助理。")

    # 🔍 讓使用者輸入要使用哪個助理（預設為小助理）
    sys_msg = input('請設定助理名稱：')
    if not sys_msg.strip():
        sys_msg = '小助理'

    # 🔍 建立對話記憶與提示
    chat_prompt, memory = create_chain(sys_msg)

    # 🔍 顯示這位助理目前的摘要記憶（若有的話）
    print(f"\n🔁 歡迎回來，{sys_msg}！")
    summary_data = memory.load_memory_variables({}).get("history", [])
    if summary_data and isinstance(summary_data, list):
        summary_text = summary_data[0].content
        print(f"🧠 上次的摘要記憶如下：\n{summary_text}\n")
    else:
        print("🧠 尚無記憶紀錄，這是你們的第一次對話。\n")

    # ✅ 組合成會話鏈（包含記憶、提示、模型），並串接 LCEL 元件
    chain = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=chat_prompt,
        verbose=True
    ) | RunnableLambda(lambda x: {"response": x['response']}) | itemgetter("response")

    print()

    # ✅ 使用者開始對話（可用 /記憶、/歷史 指令）
    while True:
        msg = input("我說：")
        if not msg.strip():
            break

        # 🔍 若輸入 /記憶，強制重新摘要對話內容
        if msg.strip() == "/記憶":
            messages = memory.chat_memory.messages
            if messages:
                summary_text = summarize_existing_history(messages)
                memory.moving_summary_buffer = summary_text  # 更新摘要記憶
                print(f"\n🧠 已更新摘要記憶內容如下：\n{summary_text}\n")
            else:
                print("\n🧠 (目前尚無摘要記憶)\n")
            continue

        # 🔍 若輸入 /歷史，顯示所有歷史對話訊息
        if msg.strip() == "/歷史":
            print("\n🗂️ SQLite 對話歷史：")
            for message in memory.chat_memory.messages:
                role = "🧑‍🦱 使用者" if message.type == "human" else "🤖 助理"
                print(f"{role}：{message.content}")
            print()
            continue

        # 🔍 一般對話流程：傳入輸入訊息，取得 AI 回應
        response = chain.invoke({"input": msg})
        print(f'{sys_msg}：{response}\n')
