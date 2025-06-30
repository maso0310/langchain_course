# 匯入 LangChain 所需的元件
# PromptTemplate：負責生成提示詞（prompt）
# StrOutputParser：將模型輸出的內容轉為字串
# Ollama：串接本地模型（例如 gemma）
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

# 🦙 載入 Ollama 本地模型
# 使用 gemma 模型（gemma3 是你自定義下載的名稱）
# 模型名稱要與你用 `ollama pull` 或 `ollama run` 的名稱一致
llm = Ollama(model="gemma3")

# 💬 建立提示模板 PromptTemplate
# LangChain 的提示模板支援使用變數嵌入（例如 {question}）
# 這裡設計固定開頭「請用繁體中文回答：」，因為 gemma 對英文回應不穩定
prompt = PromptTemplate.from_template("請用繁體中文回答：{question}")

# 🔗 建立 Chain（處理鏈）
# LangChain 支援將 prompt、模型、輸出處理等模組串接成資料流管線
# 這裡透過 `|` 運算子依序串接三個步驟：
# 1. 將使用者問題代入 prompt
# 2. 使用 llm 生成模型回應
# 3. 將模型輸出轉成純文字格式
chain = prompt | llm | StrOutputParser()

# ▶️ 執行對話流程
# 使用 `invoke()` 方法呼叫整個 Chain，並傳入變數（對應到 prompt 裡的 {question}）
# LangChain 會自動把這個變數嵌入模板、送給模型、再處理模型輸出
response = chain.invoke({"question": "LangChain 有哪些實際應用？"})

# 🖨️ 印出模型回應
print(response)
