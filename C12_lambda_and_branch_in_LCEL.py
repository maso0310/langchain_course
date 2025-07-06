from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnableLambda, RunnableBranch

llm = OllamaLLM(model="gemma3")
parser = StrOutputParser()

# 任務分類器：依關鍵字分類
task_classifier = RunnableLambda(lambda x: {
    "question": x["question"].strip(),
    "task": (
        "translate" if any(k in x["question"] for k in ["翻譯", "英文", "translate"])
        else "explain" if any(k in x["question"] for k in ["解釋", "意思", "代表"])
        else "default"
    )
})

# 各類任務 Prompt 模板
translate_prompt = ChatPromptTemplate.from_messages([("system", "你是翻譯助手，只輸出英文翻譯結果。"), ("human", "{question}")])
explain_prompt = ChatPromptTemplate.from_messages([("system", "你是知識助理，用繁體中文解釋。"), ("human", "{question}")])
default_prompt = ChatPromptTemplate.from_messages([("system", "你是 AI 助理，請用繁體中文回應。"), ("human", "{question}")])

# 任務對應處理鏈
translate_chain = translate_prompt | llm | parser
explain_chain = explain_prompt | llm | parser
default_chain = default_prompt | llm | parser

# 分支處理邏輯
task_router = RunnableBranch(
    (lambda x: x["task"] == "translate", translate_chain),
    (lambda x: x["task"] == "explain", explain_chain),
    default_chain
)

# 總流程：分類 → 分支
chain = task_classifier | task_router

# 測試
print("【翻譯例子】", chain.invoke({"question": "幫我翻譯這句話：我很喜歡打電動"}))
print("【解釋例子】", chain.invoke({"question": "請解釋一下什麼是量子糾纏"}))
print("【其他例子】", chain.invoke({"question": "你推薦什麼電影？"}))
