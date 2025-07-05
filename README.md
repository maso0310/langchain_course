# LangChain + Ollama 本地模型實戰範例

本專案收錄了一系列 LangChain 的實用範例，旨在展示如何結合本地大型語言模型（透過 Ollama 執行 `gemma`）來建構多樣化的 AI 應用。每個範例都專注於一個特定的 LangChain核心功能。

## 🚀 環境設定

在執行任何範例之前，請先完成以下設定：

### 1. 安裝 Ollama 並下載模型

請先參考 Ollama 官方網站 的說明進行安裝。安裝完成後，執行以下指令下載本專案使用的模型：

```bash
ollama pull gemma3
```

### 2. 安裝 Python 依賴套件

建議建立一個虛擬環境，並透過 `pip` 安裝所有必要的套件：

```bash
pip install langchain langchain-community langchain-core langchain-ollama ollama chromadb pydantic
```

## 📂 範例程式碼說明

以下是各個範例檔案的詳細說明：

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
| `C10_movie_information.py`             | **平行處理鏈**：學習使用 `RunnableParallel`，讓多個獨立的查詢鏈可以同時執行，最後將結果合併，提高效率。                               |

## ▶️ 如何執行

1.  請確認已完成上述的 **環境設定**。
2.  `C07` 和 `C08` 範例需要一個名為 `reference.txt` 的參考文件在同一個目錄下。
3.  `C06` 範例會在本機建立、讀取、修改 `note.txt` 檔案。
4.  在終端機中，使用 `python` 指令執行您想測試的檔案：

```bash
# 執行快速入門範例
python C01_quick_start.py

# 執行 RAG 文件問答範例
python C07_rag_doc_qa.py
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

