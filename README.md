# 多益單字 RAG 練習系統

這是一個使用 Streamlit、Chroma、sentence-transformers 與 Ollama 建立的多益單字學習工具。使用者可以輸入或上傳自己的單字資料，系統會建立本地單字知識庫，並提供單字問答與 TOEIC Reading Part 5 風格的句子填空練習題。

題目由系統根據使用者建立的單字庫生成，非 TOEIC 官方題目。

## 功能特色

- 建立自己的多益單字庫
- 支援直接輸入單字與上傳 txt 檔
- 自動排除已加入過的重複單字
- 使用 Chroma 建立本地向量資料庫
- 使用 RAG 回答單字相關問題
- 支援精準查字
- 生成 TOEIC Reading Part 5 風格句子填空題
- 測驗題提供 A-D 選項與互動作答
- 所有回答與解析以繁體中文呈現

## 技術架構

- Python
- Streamlit
- ChromaDB
- sentence-transformers
- Ollama local LLM
- pandas

## 安裝方式

1. 建立虛擬環境

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. 安裝套件

```bash
pip install -r requirements.txt
```

3. 安裝並啟動 Ollama

請先安裝 Ollama，並下載模型：

```bash
ollama pull llama3.2:3b
```

確認 Ollama 服務正在執行：

```bash
ollama serve
```

4. 啟動 Streamlit

```bash
streamlit run app.py
```

## 環境變數

系統預設使用 Ollama 模型 `llama3.2:3b`。如果想改用其他模型，可以建立 `.env`：

```env
OLLAMA_MODEL=llama3.2:3b
OLLAMA_TIMEOUT=120
```

## 使用方式

### 1. 建立單字庫

可以直接在輸入框輸入單字：

```text
invoice=n. 發票
receipt=n. 收據
applicant=n. 申請者
reschedule=v. 重新安排時間
abide by=v. 遵守
```

格式為：

```text
英文單字=詞性. 中文意思
```

也可以上傳 txt 檔，每一行放一個單字。

### 2. 單字問答

可以詢問單字意思或請系統整理相關單字，例如：

```text
punctuality 是什麼意思？
請整理 accounting 相關單字。
請列出 finance 相關的單字。
```

如果問題是明確查詢某個單字，系統會優先從單字庫精準查字；如果是整理、分類或說明類問題，則會使用 RAG 從單字庫中檢索相關資料後回答。

### 3. 多益練習題

系統會根據單字庫產生 TOEIC Reading Part 5 風格的句子填空題。每題包含：

- 一個英文句子空格
- A~D 四個選項
- 互動式選擇答案
- 作答後顯示正確答案與繁體中文解析

## 測驗生成設計

本系統的測驗生成功能採用 TOEIC Reading Part 5 句子填空題型。系統不使用 TOEIC 官方題目或第三方教材題目。

每一題皆由本地 LLM 根據 RAG 檢索到的單字資料生成，僅作為單字練習用途，並非 TOEIC 官方題目。

為了提高題目品質，系統會先用 Python 建立固定出題計畫，再交給 LLM 生成題幹與解析。出題計畫會盡量避免：

- 同一份測驗中重複使用相同選項
- 多次產生相同 target word
- 跨次測驗重複使用已出現過的單字
- 產生定義配對題，而非句子填空題

## 專案檔案

```text
app.py          Streamlit 介面
rag_core.py     單字解析、向量庫、RAG、測驗生成邏輯
prompts.py      LLM prompt 設定
requirements.txt
local_data/     本地單字資料
chroma_db/      Chroma 向量資料庫，執行後產生
```

## 注意事項

- 本專案使用本地 Ollama，不需要付費 API。
- 第一次建立單字庫時會下載或載入 sentence-transformers 模型，可能需要一些時間。
- 測驗題為系統生成，適合單字練習，不代表 TOEIC 官方題目。
- 如果題目品質不理想，可以增加單字庫內容或使用較廣的練習主題。
