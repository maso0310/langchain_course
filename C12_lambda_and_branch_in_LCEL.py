from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnableLambda, RunnableBranch

# 建立基礎元件
llm = OllamaLLM(model="gemma3")
parser = StrOutputParser()

# 前處理：從原始輸入中提取文字欄位
input_cleaner = RunnableLambda(lambda x: {"text": x["question"].strip()})

# 建立不同任務的 prompt 模板
translate_prompt = ChatPromptTemplate.from_template(
    "請你直接翻譯以下文字為英文，只輸出翻譯後的一句，**不要多種翻譯、不要說明、不要修飾語氣也不要將問句包含在內**：{text}"
)
explain_prompt = ChatPromptTemplate.from_template("請解釋：{text}")
default_prompt = ChatPromptTemplate.from_template("{text}")

# 組合各任務處理鏈（Prompt → LLM → Parser）
translate_chain = translate_prompt | llm | parser
explain_chain = explain_prompt | llm | parser
default_chain = default_prompt | llm | parser

# 建立條件分支機制：根據問題內容選擇不同任務
task_router = RunnableBranch(
    (lambda x: "翻譯" in x["text"], translate_chain),
    (lambda x: "解釋" in x["text"], explain_chain),
    default_chain
)

# 全流程組合：清理輸入 → 分流執行 → 回傳結果
chain = input_cleaner | task_router

# 測試輸入
print("【翻譯例子】")
print(chain.invoke({"question": "請幫我翻譯：我喜歡吃水果"}))

print("\n【解釋例子】")
print(chain.invoke({"question": "可以幫我解釋什麼是函數嗎？"}))

print("\n【其他例子】")
print(chain.invoke({"question": "請問什麼光源最傷眼睛？"}))
