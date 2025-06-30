from langchain_community.llms import Ollama
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory
from langchain_core.prompts import PromptTemplate

# 🦙 指定使用的模型
llm = Ollama(model="gemma3")

# 🧠 建立摘要式記憶體（會自動產生總結）
memory = ConversationSummaryMemory(llm=llm, return_messages=True)

# ✨ 自訂提示模板：一樣要求使用繁體中文回答
custom_prompt = PromptTemplate.from_template("""請使用繁體中文回答以下的問題。
{history}
人類：{input}
AI：""")

# 💬 建立對話鏈，串接記憶體與提示模板
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=custom_prompt,
    verbose=True
)

# 🗣️ 模擬對話：進行三輪問答
conversation.predict(input="你好，可以幫我規劃一下台南三日旅遊行程嗎？")
conversation.predict(input="我想要第二天去關子嶺泡溫泉，這樣安排合理嗎？")
response = conversation.predict(input="幫我簡單總結一下目前的行程規劃")

print("總結回應：", response)
