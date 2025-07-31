# LangChain x Ollama 繁體中文實戰範例

本專案包含一系列使用 LangChain 框架與 Ollama 本地大型語言模型（LLM）的 Python 實作範例。每個範例都專注於一個特定的 LangChain 功能，從基礎的處理鏈（Chain）到複雜的智慧代理人（Agent）應用。

## 環境設定

1.  **安裝 Ollama**：請先至 Ollama 官網 下載並安裝對應您作業系統的程式。

2.  **下載模型**：本專案主要使用以下幾個模型，請透過終端機執行指令下載：
    ```bash
    # 主要的語言模型
    ollama pull gemma3
    ollama pull llama3
    ollama pull mistral

    # 用於文字嵌入（Embedding）的模型
    ollama pull nomic-embed-text
    ```

3.  **安裝 Python 套件**：
    ```bash
    pip install langchain langchain-community langchain-core langchain-text-splitters langchain-ollama chromadb pydantic pymupdf duckduckgo-search requests
    ```

4.  **準備參考文件**：部分範例會讀取本地文件，請在專案根目錄下建立對應檔案：
    *   `reference.txt`：用於 `C07` 和 `C08`，請填入任意中文文字內容。
    *   `Hakka.pdf`：用於 `C19`，請準備一份 PDF 檔案。

---

## 範例總覽

| 檔案名稱                               | 主要功能與展示概念                                                                                                                             |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `C01_quick_start.py`                   | **基礎入門**：學習 LangChain 的核心概念，包括 `PromptTemplate`、`LLM` 和 `OutputParser`，並使用 LCEL 的 `|` 符號串接成一個基本的處理鏈。         |
| `C02_conversation_memory.py`           | **對話記憶**：介紹 `ConversationChain` 與 `ConversationBufferMemory`，讓模型能夠記住先前的對話內容，實現有上下文的問答。                     |
| `C03_conversation_summary_memory.py`   | **摘要式記憶**：展示 `ConversationSummaryMemory`，它會自動將對話歷史總結成摘要，適合處理長篇對話以節省 token。                         |
| `C04_agent.py`                         | **單一工具 Agent**：學習如何建立一個 Agent，並賦予它一個自定義工具（平方根計算器），讓模型能根據問題決定是否使用該工具。                   |
| `C05_agent_multi_tool.py`              | **多工具 Agent**：擴充 Agent 的能力，讓它能同時擁有多個工具（計算器、模擬搜尋），並根據任務需求自主選擇最適合的工具。                   |
| `C06_agent_note_tool.py`               | **具備記憶與檔案操作的 Agent**：建立一個更複雜的 Agent，它不僅擁有對話記憶，還能操作本地檔案（建立、讀取、追加、修改筆記）。         |
| `C07_rag_doc_qa.py`                    | **RAG 文件問答系統**：展示如何建構一個完整的 RAG 流程。從讀取本地文件、切割、向量化（使用 ChromaDB），到建立一個能根據文件內容回答問題的 QA 鏈。 |
| `C08_rag_tool_agent.py`                | **RAG + Agent 整合應用**：將 RAG 問答系統包裝成一個 Tool，再與其他工具（如計算機）一起交給 Agent 管理。Agent 會根據問題的性質，決定是查詢文件還是進行計算。 |
| `C09_career_assistant.py`              | **結構化輸出**：介紹如何使用 `PydanticOutputParser`，強制模型回傳符合特定 JSON 格式的結構化資料，非常適合需要穩定資料格式的應用。     |
| `C10_movie_information.py`             | **平行處理鏈**：學習使用 `RunnableParallel`，讓多個獨立的查詢鏈可以同時執行，最後將結果合併，提高效率。                                 |
| `C11_attrgetter_and_itemgetter.py`     | **LCEL 資料流操作**：展示如何使用 `itemgetter` 和 `attrgetter` 在處理鏈中精準地擷取與傳遞特定資料，優化複雜的資料流。                   |
| `C12_lambda_and_branch_in_LCEL.py`     | **LCEL 條件分支**：學習使用 `RunnableLambda` 進行任務分類，並透過 `RunnableBranch` 根據分類結果將請求導向不同的處理鏈。                 |
| `C13_configurable_alternatives.py`     | **動態配置處理鏈**：展示如何使用 `with_config` 在執行時動態切換不同的 Prompt 或 LLM 模型，增加處理鏈的彈性。                           |
| `C14_conversation_RunnableWithMessageHistory.py` | **新式對話記憶**：介紹 `RunnableWithMessageHistory`，這是 LCEL 中管理對話歷史的現代化方法，提供更好的整合性與彈性。                 |
| `C15_conversation_history_to_db.py`    | **對話紀錄存入資料庫**：將聊天歷史儲存於 SQLite 資料庫，並結合摘要記憶，實現可長期保存且高效的對話機器人。                             |
| `C16_chain_stream.py`                  | **串流輸出**：展示如何讓 LLM 的回應以串流（Stream）方式逐字輸出，即時顯示結果，大幅提升使用者體驗。                                   |
| `C17_web_search_with_langchain.py`     | **Agent 網路搜尋**：整合 `DuckDuckGo` 搜尋工具，讓 Agent 能夠上網查詢即時資訊，回答訓練資料中沒有的時事問題。                         |
| `C18_weather_API.py`                   | **Agent 串接 API**：使用 `StructuredTool` 建立一個能呼叫外部天氣 API 的工具，讓 Agent 能夠獲取並整理結構化的即時數據。                 |
| `C19_PDF_loader.py`                    | **PDF 文件問答**：展示如何使用 `PyMuPDFLoader` 載入並處理 PDF 檔案，並建立一個針對 PDF 內容的 RAG 問答系統。                         |


```bash
# 執行快速入門範例
python C01_quick_start.py
```

## 💡 核心概念總結

本專案涵蓋了 LangChain 的多個核心模組與設計模式：

-   **Models (LLMs & Embeddings)**: 串接 Ollama 本地模型作為語言模型與嵌入模型。
-   **Prompts**: 使用 `PromptTemplate` 和 `ChatPromptTemplate` 進行結構化的提示詞工程。
-   **Chains (LCEL)**: 透過 LangChain Expression Language (LCEL) 的 `|` 語法，彈性地組合各種元件。
-   **Memory**: 利用 `ConversationBufferMemory` 和 `ConversationSummaryMemory` 讓對話具有上下文記憶。
-   **Retrieval-Augmented Generation (RAG)**: 結合外部文件（`reference.txt`）與向量資料庫（`Chroma`），讓模型能回答其訓練資料中沒有的特定知識。
-   **Agents & Tools**: 賦予 LLM 使用外部工具（如計算機、檔案系統、RAG系統）的能力，使其能完成更複雜的任務。
-   **Output Parsers**: 使用 `StrOutputParser` 和 `PydanticOutputParser` 來解析模型的輸出，使其成為純文字或結構化的資料格式。
-   **Runnables**: 運用 `RunnableParallel` 等元件來平行化處理任務，提升執行效率。

