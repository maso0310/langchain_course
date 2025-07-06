from langchain_core.runnables import ConfigurableField
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_ollama import OllamaLLM

# 定義三種 Prompt 模板
prompt = PromptTemplate.from_template("用繁體中文像朋友聊天回應我這段話：{topic}").configurable_alternatives(
    ConfigurableField(id="prompt", name="Prompt", description="提詞組合"),
    default_key="friendly",
    formal=PromptTemplate.from_template("用繁體中文正式地回應我這段話：{topic}"),
    wiseman=ChatPromptTemplate.from_messages([
        ("system", "你是一個睿智的老人"),
        ("human", "用說故事的方式，用繁體中文回應我的這段話：{topic}")
    ]),
)

# 定義三種 Ollama 模型（gemma3、llama3 和 mistral）
llm = OllamaLLM(model="gemma3").configurable_alternatives(
    ConfigurableField(id="llm", name="LLM", description="語言模型"),
    default_key="gemma3",
    llama3=OllamaLLM(model="llama3"),
    mistral=OllamaLLM(model="mistral")
)

# 串接整個 chain：Prompt → LLM → Parser
chain = prompt | llm | StrOutputParser()

# 測試預設組合（friendly + gemma3）
print("🟢 預設組合（friendly + gemma3）：", chain.invoke({"topic": "請問如何規劃人生方向？"}))

# 測試：改成 formal + llama3
chain_custom = chain.with_config(configurable={
    "prompt": "formal",
    "llm": "llama3"
})
print("\n🔵 切換組合（ formal + llama3 ）：", chain_custom.invoke({"topic": "請問如何規劃人生方向？"}))

# 測試：改成 wiseman + mistral
chain_custom = chain.with_config(configurable={
    "prompt": "wiseman",
    "llm": "mistral"
})
print("\n🔵 切換組合（ wiseman + mistral ）：", chain_custom.invoke({"topic": "請問如何規劃人生方向？"}))
