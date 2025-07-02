from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnableParallel

# ✅ 建立模型與字串解析器
llm = OllamaLLM(model="gemma3")
str_parser = StrOutputParser()

# ✅ 各個子任務的 PromptTemplate + Chain

# 查詢電影上映年份
year_prompt = ChatPromptTemplate.from_template("請問電影《{movie}》是在哪一年上映的？")
find_year_chain = year_prompt | llm | str_parser

# 查詢導演
director_prompt = ChatPromptTemplate.from_template("請問電影《{movie}》的導演是誰？")
find_director_chain = director_prompt | llm | str_parser

# 查詢主要演員
actors_prompt = ChatPromptTemplate.from_template("電影《{movie}》的主要演員有哪些？")
find_actors_chain = actors_prompt | llm | str_parser

# 查詢電影類型
genre_prompt = ChatPromptTemplate.from_template("請問電影《{movie}》屬於哪一種類型？")
find_genre_chain = genre_prompt | llm | str_parser

# 查詢簡短劇情
summary_prompt = ChatPromptTemplate.from_template("請簡要介紹電影《{movie}》的劇情")
find_summary_chain = summary_prompt | llm | str_parser

# ✅ 合併為一個 Parallel Chain
movie_info_chain = RunnableParallel(
    year=find_year_chain,
    director=find_director_chain,
    actors=find_actors_chain,
    genre=find_genre_chain,
    summary=find_summary_chain,
)

# ✅ 測試查詢
result = movie_info_chain.invoke({"movie": "天能"})
for key, value in result.items():
    print(f"{key.upper()}:\n{value}\n")
