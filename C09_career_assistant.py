import json
from typing import List
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

# 定義求職建議格式
class JobAdvice(BaseModel):
    suggested_roles: List[str] = Field(description="建議的職位名稱")
    required_skills: List[str] = Field(description="需要具備的技能")
    learning_resources: List[str] = Field(description="推薦學習資源")
    salary_estimate: str = Field(description="預估的薪資範圍")

# 建立模型
llm = OllamaLLM(model="gemma3")

# 建立輸出內容解析器（將模型回應解析為 Pydantic 結構）
parser = PydanticOutputParser(pydantic_object=JobAdvice)
format_instructions = parser.get_format_instructions()

# 建立 Prompt 模板，包含 system 訊息與使用者查詢，並嵌入格式說明
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位職涯顧問，請根據使用者興趣給出個人化的求職建議，並以繁體中文回應。\n{format_instructions}"),
    ("human", "{query}")
])
final_prompt = prompt.partial(format_instructions=format_instructions)

# 將使用者查詢代入 Prompt，生成完整提示內容
query = "我對利用物聯網、無人機、網路應用程式開發與AI資料分析技術應用於農業發展有興趣"
formatted_prompt = final_prompt.invoke({"query": query})

# # 建立 LangChain 流程鏈：Prompt → 模型 → 輸出解析
chain = final_prompt | llm | parser
response = chain.invoke(formatted_prompt)
print(json.dumps(response.dict(), ensure_ascii=False, indent=2))
