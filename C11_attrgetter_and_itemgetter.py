from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnableParallel

llm = OllamaLLM(model="gemma3")
# 建立模型與文字解析器
str_parser = StrOutputParser()

# 建立針對「國家」的提詞模板與處理流程
country_template = ChatPromptTemplate.from_template('{city} 位於哪一個國家？')
find_country_chain = country_template | llm | str_parser

# 建立針對「語言」的提詞模板與處理流程
lang_template = ChatPromptTemplate.from_template('在 {city} 講哪一種語言？')
find_lang_chain = lang_template | llm | str_parser

# 建立摘要用的提示模板，將前兩個鏈的輸出填入 {country} 和 {lang}
summary_template = ChatPromptTemplate.from_template('{country}{lang}')

# 建立一條新的鏈：先執行兩條鏈（國家與語言），再把結果帶入 summary_template 組成提示
summary_chain = (
    {
        'country': find_country_chain,
        'lang': find_lang_chain
    }
    | summary_template
)

# 引入工具：從 PromptTemplate 中提取 messages 並取出第一條
from operator import attrgetter, itemgetter
get_messages = attrgetter('messages')
get_first_item = itemgetter(0)

# 將組合後的提示鏈（summary_chain）擷取出文字形式，方便檢查 Prompt 長相
summary = (summary_chain
           | get_messages
           | get_first_item
           | str_parser)

print(summary.invoke({'city': '釜山'}))
