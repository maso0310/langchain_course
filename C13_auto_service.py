from langchain_ollama import OllamaLLM
from langchain.agents import Tool, initialize_agent
import pandas as pd

# 載入菜單
menu_df = pd.read_csv("menu.csv")

# 建立訂單記憶
current_order = []

# 各種功能工具（以 Tool 包裝）

def query_menu_func(category: str = "") -> str:
    df = menu_df[menu_df["品項類別"] == category] if category else menu_df
    return df.to_string(index=False)

def add_to_order_func(item: str, spec: str = "", qty: int = 1) -> str:
    if not item:
        return "請提供要加入的品項名稱。"

    row = menu_df[menu_df["品項名稱"] == item]
    if row.empty:
        return f"找不到品項：{item}"

    # 自動填預設規格（如果只有單一選項）
    valid_specs = row["可選規格"].values[0].split("/")
    if not spec or spec not in valid_specs:
        return f"{item} 的可選規格有：{'/'.join(valid_specs)}，請指定其中一種。"

    try:
        qty = int(qty)
    except:
        return "數量請輸入有效的整數。"

    price = int(row["價格"].values[0])
    subtotal = price * qty
    current_order.append({
        "品項名稱": item,
        "規格": spec,
        "數量": qty,
        "單價": price,
        "小計": subtotal
    })
    return f"✅ 已加入：{item}（{spec}）x{qty}，小計{subtotal}元"


def show_order_func() -> str:
    if not current_order:
        return "目前尚未點餐。"
    total = sum(i["小計"] for i in current_order)
    lines = [f"- {i['品項名稱']}（{i['規格']}）x{i['數量']} = {i['小計']}元" for i in current_order]
    return "\n".join(lines) + f"\n總金額：{total}元"

def clear_order_func() -> str:
    current_order.clear()
    return "🗑️ 已清空訂單！"

def checkout_func() -> dict:
    total = sum(i["小計"] for i in current_order)
    return {
        "訂單明細": current_order,
        "總金額": total
    }

# 包裝成 Tool
tools = [
    Tool.from_function(
        name="查詢菜單",
        func=query_menu_func,
        description="查詢指定類別或完整的菜單內容，適合回答使用者關於有什麼品項的問題。"
    ),
    Tool.from_function(
        name="新增品項",
        func=add_to_order_func,
        description="將指定的品項與數量加入訂單中，格式如 '我要一份大麥克、兩杯可樂。'"
    ),
    Tool.from_function(
        name="查看訂單",
        func=show_order_func,
        description="查看目前已加入訂單的所有內容與總金額。"
    ),
    Tool.from_function(
        name="清除訂單",
        func=clear_order_func,
        description="清除目前的點餐內容，讓使用者重新開始點餐。"
    ),
    Tool.from_function(
        name="結帳",
        func=checkout_func,
        description="輸出最終的訂單資訊，包含明細與總金額，適合在點餐完成後呼叫。"
    ),
]

# 使用 Gemma 模型
llm = OllamaLLM(model="gemma3")

# 建立 Agent Executor（不需 create_tool_calling_agent）
agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type="zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True,
    agent_kwargs={
        "prefix": "你是一位中文語言的智慧點餐助手，請用繁體中文與使用者互動並使用繁體中文呼叫工具。如果找不到可用工具請直接回答",
        "format_instructions": "請使用中文描述你的推理過程與行動。",
    }
)

# 主程式互動介面
if __name__ == "__main__":
    print("歡迎來到智慧點餐系統！（輸入 exit 離開）")
    while True:
        user_input = input("你：").strip()
        if user_input.lower() == "exit":
            break
        result = agent_executor.invoke(user_input)
        print("AI：", result)
