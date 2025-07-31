import requests
from langchain_ollama import OllamaLLM
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 定義輸入格式（使用 Pydantic）
class Weather(BaseModel):
    city: str = Field(description="台灣縣市, 使用繁體中文")  # 使用者需要輸入的城市名稱

# 定義取得天氣的函式，透過 Google Apps Script API 擷取資料
def get_weather(city: str):
    API_URL = f"https://script.google.com/macros/s/AKfycbwIZI5Ha9vOhq3fACslYwnhFPM8pM3Dlb5R7l8aorSTyQiO8JVG56G_rYr60YbdvNs4/exec?city={city}"
    response = requests.get(API_URL)  # 送出 HTTP GET 請求
    return response.json()  # 回傳 JSON 格式的天氣資料

# ✅ 初始化本地 Ollama 模型（例如 gemma3）
llm = OllamaLLM(model="gemma3")

# ✅ 設定字串解析器（將模型輸出轉為純文字）
str_parser = StrOutputParser()

# ✅ 註冊工具為 StructuredTool，支援自動參數推理與驗證
weather_data = StructuredTool.from_function(
    func=get_weather,              # 綁定的函式
    name="weather-data",          # 工具名稱
    description="得到台灣縣市天氣資料",  # 工具用途簡介
    args_schema=Weather           # 指定參數格式
)

# ✅ 定義提示模板，告訴 LLM 如何處理天氣資料
weather_template = "請彙整縣市一周的天氣資訊{weather}並回答天氣資訊"
weather_prompt = ChatPromptTemplate.from_template(weather_template)

# ✅ 串接整體 Chain
# 使用者輸入城市 → 工具取得天氣資料 → 套用提示模板 → 使用 LLM 處理 → 解析為文字
chain = ({"weather": weather_data}     # 使用 tool 回傳資料後代入 prompt
         | weather_prompt              # 插入到 prompt 模板中
         | llm                         # 交由本地 LLM 產生回應
         | str_parser)                # 輸出純文字結果

# ✅ 呼叫整體 Chain，這裡以「新北市」為例
print(chain.invoke("臺中市"))
