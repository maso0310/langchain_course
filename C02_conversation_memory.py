from langchain_community.llms import Ollama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

# 🦙 使用 gemma3 模型
llm = Ollama(model="gemma3")

# 🧠 建立記憶體，記錄上下文對話
memory = ConversationBufferMemory(return_messages=True)

# ✨ 自訂提示模板：要求用繁體中文回答
custom_prompt = PromptTemplate.from_template("""
請使用繁體中文回答以下的問題。
{history} 
人類：{input} 
AI：""")

# 💬 建立對話鏈，指定使用自訂提示模板
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=custom_prompt,
    verbose=True
)

# 🗣️ 執行兩輪問答
response1 = conversation.predict(input="請問你是誰？")
print("第一輪回應：", response1)

response2 = conversation.predict(input="你剛剛說你是 AI，那你能做什麼？")
print("第二輪回應：", response2)
