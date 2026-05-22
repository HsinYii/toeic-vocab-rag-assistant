import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union

import chromadb
import pandas as pd
from dotenv import load_dotenv
# from openai import OpenAI

from prompts import RAG_QA_PROMPT, QUIZ_GENERATION_PROMPT

import requests
from sentence_transformers import SentenceTransformer

load_dotenv()

try:
    from opencc import OpenCC

    _OPENCC = OpenCC("s2twp")
except Exception:
    _OPENCC = None

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "toeic_vocab"

STANDARD_COLUMNS = [
    "word",
    "part_of_speech",
    "zh_meaning",
    "en_definition",
    "source_file",
    "raw_text",
    "line_no",
]


# def get_openai_client() -> OpenAI:
#     api_key = os.getenv("OPENAI_API_KEY")
#     if not api_key:
#         raise RuntimeError("找不到 OPENAI_API_KEY，請先在 .env 設定。")
#     return OpenAI(api_key=api_key)


def decode_bytes(data: bytes) -> str:
    """嘗試用常見編碼解碼 txt 檔。"""
    encodings = ["utf-8-sig", "utf-8", "cp950", "big5", "latin-1"]
    for enc in encodings:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def read_uploaded_text(uploaded_file) -> str:
    return decode_bytes(uploaded_file.getvalue())


def read_path_text(path: Union[str, Path]) -> str:
    return decode_bytes(Path(path).read_bytes())


def contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf" for ch in text)


SIMPLIFIED_TO_TRADITIONAL = str.maketrans(
    {
        "个": "個",
        "为": "為",
        "会": "會",
        "这": "這",
        "现": "現",
        "习": "習",
        "词": "詞",
        "义": "義",
        "选": "選",
        "项": "項",
        "题": "題",
        "语": "語",
        "说": "說",
        "明": "明",
        "释": "釋",
        "对": "對",
        "应": "應",
        "单": "單",
        "复": "複",
        "确": "確",
        "认": "認",
        "让": "讓",
        "时": "時",
        "间": "間",
        "发": "發",
        "与": "與",
        "关": "關",
        "联": "聯",
        "资": "資",
        "讯": "訊",
        "产": "產",
        "业": "業",
        "务": "務",
        "员": "員",
        "习": "習",
        "测": "測",
        "验": "驗",
        "数": "數",
        "据": "據",
        "库": "庫",
        "构": "構",
        "练": "練",
        "学": "學",
        "习": "習",
        "进": "進",
        "过": "過",
        "还": "還",
        "没": "沒",
        "经": "經",
        "请": "請",
        "错": "錯",
        "该": "該",
        "将": "將",
        "见": "見",
        "别": "別",
        "类": "類",
        "组": "組",
        "读": "讀",
        "写": "寫",
        "内": "內",
        "码": "碼",
        "块": "塊",
        "额": "額",
        "显": "顯",
        "示": "示",
        "标": "標",
        "记": "記",
        "换": "換",
        "边": "邊",
        "辑": "輯",
        "审": "審",
        "导": "導",
        "输": "輸",
        "历": "歷",
        "万": "萬",
        "两": "兩",
        "处": "處",
        "无": "無",
        "从": "從",
        "后": "後",
        "只": "只",
        "体": "體",
        "苹": "蘋",
        "国": "國",
        "应": "應",
        "买": "買",
        "卖": "賣",
        "价": "價",
        "门": "門",
        "专": "專",
        "书": "書",
        "车": "車",
        "电": "電",
        "话": "話",
        "网": "網",
        "页": "頁",
        "开": "開",
        "闭": "閉",
        "广": "廣",
        "办": "辦",
        "证": "證",
        "么": "麼",
        "吗": "嗎",
        "刚": "剛",
        "长": "長",
        "条": "條",
        "东": "東",
        "圆": "圓",
        "旧": "舊",
        "新": "新",
        "爱": "愛",
        "乐": "樂",
    }
)


def to_traditional_chinese(text: str) -> str:
    if _OPENCC is not None:
        return _OPENCC.convert(text)
    return text.translate(SIMPLIFIED_TO_TRADITIONAL)


def normalize_word(word: str) -> str:
    word = word.strip().lstrip("\ufeff")
    word = re.sub(r"\s+", " ", word)
    return word


def normalize_pos(pos: str) -> str:
    if not pos:
        return ""

    pos = pos.lower().strip().strip(".")
    pos = pos.replace(" ", "")

    mapping = {
        "n": "noun",
        "v": "verb",
        "vt": "verb",
        "vi": "verb",
        "adj": "adjective",
        "a": "adjective",
        "adv": "adverb",
        "prep": "preposition",
        "pre": "preposition",
        "conj": "conjunction",
        "pron": "pronoun",
        "abbr": "abbreviation",
    }

    if "/" in pos:
        parts = [mapping.get(p.strip(), p.strip()) for p in pos.split("/") if p.strip()]
        return "/".join(parts)

    return mapping.get(pos, pos)


def split_pos_and_meaning(rest: str) -> Tuple[str, str]:
    """
    將 v. 遵守,信守 或 n. skill, talent 拆成 pos / meaning。
    若格式不符，就把整段當 meaning。
    """
    rest = rest.strip()
    match = re.match(r"^([A-Za-z/,&+\- ]{1,25})\.\s*(.+)$", rest)
    if match:
        pos = match.group(1).strip()
        meaning = match.group(2).strip()
        return pos, meaning
    return "", rest


def parse_vocab_text(text: str, source_file: str) -> pd.DataFrame:
    rows = []

    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip().lstrip("\ufeff")
        line = line.replace("＝", "=").replace("．", ".")
        line = to_traditional_chinese(line)

        if not line or line.startswith("#"):
            continue

        if "=" not in line:
            continue

        word_part, rest = line.split("=", 1)
        word = normalize_word(word_part)

        if not word:
            continue

        pos, meaning = split_pos_and_meaning(rest)
        pos = normalize_pos(pos)

        if contains_cjk(meaning):
            zh_meaning = to_traditional_chinese(meaning)
            en_definition = ""
        else:
            zh_meaning = ""
            en_definition = meaning

        rows.append(
            {
                "word": word,
                "part_of_speech": pos,
                "zh_meaning": zh_meaning,
                "en_definition": en_definition,
                "source_file": source_file,
                "raw_text": line,
                "line_no": line_no,
            }
        )

    return pd.DataFrame(rows, columns=STANDARD_COLUMNS)


def merge_unique(values, sep: str = "；") -> str:
    seen = []
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text or text.lower() == "nan":
            continue
        if text not in seen:
            seen.append(text)
    return sep.join(seen)


def merge_vocab_frames(frames: List[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(
            columns=[
                "word",
                "part_of_speech",
                "zh_meaning",
                "en_definition",
                "source_files",
                "raw_sources",
            ]
        )

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["word"])
    df["word"] = df["word"].astype(str).map(normalize_word)
    df = df[df["word"] != ""]
    df["word_key"] = df["word"].str.lower().str.strip()

    records = []

    for _, group in df.groupby("word_key", dropna=True):
        word = group["word"].iloc[0]

        raw_sources = []
        for _, row in group.iterrows():
            raw_sources.append(
                f"{row['source_file']}:{row['line_no']} | {row['raw_text']}"
            )

        records.append(
            {
                "word": word,
                "part_of_speech": merge_unique(group["part_of_speech"], sep="/"),
                "zh_meaning": merge_unique(group["zh_meaning"], sep="；"),
                "en_definition": merge_unique(group["en_definition"], sep="; "),
                "source_files": merge_unique(group["source_file"], sep=";"),
                "raw_sources": "\n".join(raw_sources),
            }
        )

    merged = pd.DataFrame(records)
    merged = merged.sort_values("word").reset_index(drop=True)
    return merged


def load_local_vocab_files(local_dir: str = "local_data") -> Tuple[pd.DataFrame, List[str]]:
    filenames = ["toeic1.txt", "toeicee.txt", "newtoeicl5_tw.txt"]
    frames = []
    missing = []

    for filename in filenames:
        path = Path(local_dir) / filename
        if path.exists():
            text = read_path_text(path)
            frames.append(parse_vocab_text(text, filename))
        else:
            missing.append(filename)

    if not frames:
        raise FileNotFoundError(
            f"在 {local_dir}/ 找不到任何 txt 檔，請放入 toeic1.txt、toeicee.txt、newtoeicl5_tw.txt。"
        )

    return merge_vocab_frames(frames), missing


def build_chunk(row: pd.Series) -> str:
    word = row.get("word", "")
    pos = row.get("part_of_speech", "") or "N/A"
    zh = row.get("zh_meaning", "") or "N/A"
    en = row.get("en_definition", "") or "N/A"
    sources = row.get("source_files", "") or "N/A"
    raw_sources = row.get("raw_sources", "") or "N/A"

    return f"""Word: {word}
Part of speech: {pos}
Chinese meaning: {zh}
English definition: {en}
Source files: {sources}

Raw sources:
{raw_sources}
"""


def dataframe_to_chunks(df: pd.DataFrame):
    documents = []
    metadatas = []
    ids = []

    for idx, row in df.reset_index(drop=True).iterrows():
        doc = build_chunk(row)
        doc_id = hashlib.md5(f"{row.get('word', '')}-{idx}".encode("utf-8")).hexdigest()

        documents.append(doc)
        ids.append(doc_id)
        metadatas.append(
            {
                "word": str(row.get("word", "")),
                "part_of_speech": str(row.get("part_of_speech", "")),
                "source_files": str(row.get("source_files", "")),
            }
        )

    return documents, metadatas, ids

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model

def embed_texts(texts: List[str], batch_size: int = 64) -> List[List[float]]:
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return embeddings.tolist()

def call_ollama(system_prompt: str, user_prompt: str, timeout: Union[int, None] = None) -> str:
    model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    request_timeout = timeout or int(os.getenv("OLLAMA_TIMEOUT", "120"))
    zh_tw_rule = (
        "\n\n全域輸出規則：除英文單字、英文例句、TOEIC 題幹、固定選項或必要欄位標籤外，"
        "所有回答、說明、解析、提示與補充內容都必須使用繁體中文。"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt + zh_tw_rule},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
    }

    response = requests.post(
        "http://localhost:11434/api/chat",
        json=payload,
        timeout=request_timeout,
    )

    response.raise_for_status()
    return response.json()["message"]["content"]


def get_chroma_client(persist_dir: str = CHROMA_DIR):
    return chromadb.PersistentClient(path=persist_dir)


def reset_collection(client, collection_name: str = COLLECTION_NAME):
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    return client.create_collection(
        name=collection_name,
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )


def build_vector_db(
    df: pd.DataFrame,
    persist_dir: str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> Dict:
    documents, metadatas, ids = dataframe_to_chunks(df)

    if not documents:
        raise ValueError("沒有可建立向量資料庫的單字資料。")

    client = get_chroma_client(persist_dir)
    collection = reset_collection(client, collection_name)

    batch_size = 100

    for start in range(0, len(documents), batch_size):
        batch_docs = documents[start : start + batch_size]
        batch_metas = metadatas[start : start + batch_size]
        batch_ids = ids[start : start + batch_size]
        batch_embeddings = embed_texts(batch_docs)

        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            embeddings=batch_embeddings,
            metadatas=batch_metas,
        )

    return {
        "count": len(documents),
        "persist_dir": persist_dir,
        "collection_name": collection_name,
    }


def retrieve_context(
    query: str,
    top_k: int = 5,
    persist_dir: str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> List[Dict]:
    client = get_chroma_client(persist_dir)
    collection = client.get_collection(collection_name)

    query_embedding = embed_texts([query])[0]

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    retrieved = []

    for doc, meta, dist in zip(documents, metadatas, distances):
        retrieved.append(
            {
                "document": doc,
                "metadata": meta,
                "distance": dist,
            }
        )

    return retrieved


def format_context(retrieved_results: List[Dict]) -> str:
    parts = []

    for i, item in enumerate(retrieved_results, start=1):
        parts.append(
            f"""[資料來源 {i}]
檢索距離：{item["distance"]}
metadata：{item["metadata"]}
{item["document"]}
"""
        )

    return "\n\n".join(parts)


def generate_answer(question: str, retrieved_results: List[Dict]) -> str:
    context = format_context(retrieved_results)

    user_prompt = f"""單字資料：
{context}

使用者問題：
{question}

請一律使用繁體中文回答；必要的英文單字、英文例句與詞性縮寫可以保留英文。
"""

    return to_traditional_chinese(call_ollama(RAG_QA_PROMPT, user_prompt))


def normalize_quiz_payload(parsed: Union[Dict, List]) -> Dict:
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        return {"questions": parsed}
    raise ValueError("模型回傳的 JSON 不是測驗物件或題目陣列。")


def extract_json_object(text: str) -> Dict:
    """Parse model output that should be JSON, tolerating accidental fences/text."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return normalize_quiz_payload(json.loads(cleaned))
    except json.JSONDecodeError:
        object_start = cleaned.find("{")
        array_start = cleaned.find("[")
        starts = [index for index in [object_start, array_start] if index != -1]
        start = min(starts) if starts else -1
        if start == -1:
            raise ValueError("模型沒有回傳可解析的 JSON 測驗資料。")
        try:
            parsed, _ = json.JSONDecoder().raw_decode(cleaned[start:])
            return normalize_quiz_payload(parsed)
        except json.JSONDecodeError as e:
            raise ValueError(f"模型沒有回傳可解析的 JSON 測驗資料：{e}")


def normalize_answer_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).strip(" .。").lower()


def normalize_blank_marker(text: str) -> str:
    return re.sub(r"_{4,}", "______", str(text).strip())


def clean_choice_text(text: str) -> str:
    value = str(text).strip().strip("\"'`“”‘’")
    value = re.sub(r"^[A-Da-d][\).:：、]\s*", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def normalize_option_key(key: str) -> str:
    text = str(key).strip().upper()
    match = re.search(r"\b([A-D])\b", text)
    if match:
        return match.group(1)
    return text[:1]


def normalize_answer_letter(answer: str, option_map: Dict[str, str]) -> str:
    text = str(answer).strip()
    upper_text = text.upper()

    match = re.match(r"^([A-D])(?:\b|[\).:：、])", upper_text)
    if match:
        return match.group(1)

    if upper_text in option_map:
        return upper_text

    answer_key = normalize_answer_text(text)
    for letter, option_text in option_map.items():
        if normalize_answer_text(option_text) == answer_key:
            return letter

    return upper_text[:1]


def is_english_word_or_phrase(text: str) -> bool:
    value = text.strip()
    if contains_cjk(value):
        return False
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z' -]*[A-Za-z]", value))


def is_phrase(text: str) -> bool:
    return len(text.strip().split()) > 1


def validate_quiz_style(quiz_text: str) -> List[str]:
    """
    Validate whether generated quiz roughly follows TOEIC Part 5 format.
    Returns warning messages.
    """
    warnings = []

    banned_phrases = [
        "TOEIC vocabulary term meaning",
        "the word meaning",
        "word meaning",
        "Chinese meaning",
        "中文意思",
        "意思是",
    ]

    lower_text = quiz_text.lower()

    for phrase in banned_phrases:
        if phrase.lower() in lower_text:
            warnings.append(
                f"偵測到可能像定義查詢題的文字：{phrase}"
            )

    if "______" not in quiz_text:
        warnings.append("未找到空格 `______`，Part 5 題目需要句子填空。")

    required_options = ["(A)", "(B)", "(C)", "(D)"]
    for option in required_options:
        if option not in quiz_text:
            warnings.append(f"缺少選項標籤：{option}")

    if "Answer:" not in quiz_text:
        warnings.append("缺少 `Answer:` 欄位。")

    if "Target word:" not in quiz_text:
        warnings.append("缺少 `Target word:` 欄位。")

    return warnings


def has_definition_question_style(question_text: str) -> bool:
    lower_text = question_text.lower()
    banned_patterns = [
        "the word meaning",
        "the toeic vocabulary term meaning",
        "the vocabulary term meaning",
        "chinese meaning",
        "中文意思",
        "意思是",
        "which word means",
        "which word meaning",
    ]
    return any(pattern in lower_text for pattern in banned_patterns)


def has_generic_quiz_template(question_text: str) -> bool:
    lower_text = re.sub(r"\s+", " ", question_text.lower()).strip()
    generic_patterns = [
        "the manager reviewed the ______ before approving the request",
        "the team discussed the ______ during the meeting",
        "the team discussed the ______ during the weekly business meeting",
        "the sales department prepared the ______ for the client",
        "please ______",
    ]
    return any(pattern in lower_text for pattern in generic_patterns)


def questions_to_quiz_text(questions: List[Dict]) -> str:
    blocks = []
    for index, question in enumerate(questions, start=1):
        option_lines = [
            f"({letter}) {text}"
            for letter, text in question.get("options", {}).items()
        ]
        blocks.append(
            "\n".join(
                [
                    f"Question {index}:",
                    question.get("question", ""),
                    "",
                    *option_lines,
                    "",
                    f"Answer: {question.get('answer', '')}",
                    f"Target word: {question.get('word', '')}",
                    f"Explanation: {question.get('explanation', '')}",
                    f"Source: {question.get('source', question.get('word', ''))}",
                ]
            )
        )
    return "\n\n".join(blocks)


def validate_quiz_questions_style(questions: List[Dict]) -> List[str]:
    warnings = validate_quiz_style(questions_to_quiz_text(questions))

    for index, question in enumerate(questions, start=1):
        question_text = question.get("question", "")
        blank_count = question_text.count("______")
        if blank_count != 1:
            warnings.append(
                f"第 {index} 題應該只有一個 `______` 空格，目前找到 {blank_count} 個。"
            )
        if has_definition_question_style(question_text):
            warnings.append(
                f"第 {index} 題看起來像定義配對題。"
            )
        if has_generic_quiz_template(question_text):
            warnings.append(
                f"第 {index} 題使用過於泛用的句型。"
            )

    return warnings


def normalize_quiz_questions(
    raw_quiz: Union[Dict, List],
    expected_count: int,
    used_words: Union[set, None] = None,
) -> List[Dict]:
    raw_quiz = normalize_quiz_payload(raw_quiz)
    questions = raw_quiz.get("questions", [])

    if not isinstance(questions, list) or not questions:
        raise ValueError("模型回傳的測驗資料缺少 questions。")

    if len(questions) < expected_count:
        raise ValueError(
            f"模型回傳 {len(questions)} 題，少於要求的 {expected_count} 題。"
        )

    normalized = []
    seen_questions = set()
    seen_option_sets = set()
    seen_words = set(used_words or set())
    phrase_count = 0
    max_phrase_count = max(1, expected_count // 4)

    for item in questions:
        if len(normalized) == expected_count:
            break

        if not isinstance(item, dict):
            continue

        options = item.get("options", {})
        if isinstance(options, list):
            option_map = {}
            for letter, value in zip(["A", "B", "C", "D"], options):
                option_map[letter] = clean_choice_text(value)
            options = option_map
        elif isinstance(options, dict):
            options = {
                normalize_option_key(key): clean_choice_text(value)
                for key, value in options.items()
            }
        else:
            options = {}

        option_map = {
            letter: options.get(letter, "")
            for letter in ["A", "B", "C", "D"]
            if options.get(letter, "")
        }
        answer = normalize_answer_letter(item.get("answer", ""), option_map)
        question_text = normalize_blank_marker(item.get("question", ""))
        word = clean_choice_text(
            item.get("word", "")
            or item.get("target_word", "")
            or item.get("target word", "")
            or item.get("Target word", "")
        )
        if not word and answer in option_map:
            word = option_map[answer]
        question_key = re.sub(r"\s+", " ", question_text.lower())
        word_key = normalize_answer_text(word)

        has_chinese_test_content = contains_cjk(question_text) or any(
            contains_cjk(option_text) for option_text in option_map.values()
        )
        has_invalid_option = any(
            not is_english_word_or_phrase(option_text)
            for option_text in option_map.values()
        )
        option_values = [
            normalize_answer_text(option_text) for option_text in option_map.values()
        ]
        option_set_key = tuple(sorted(option_values))
        correct_option = option_map.get(answer, "")
        answer_is_visible = bool(
            word_key and re.search(rf"\b{re.escape(word_key)}\b", question_key)
        )
        uses_phrase = is_phrase(word)

        if (
            not question_text
            or question_text.count("____") != 1
            or len(option_map) != 4
            or len(set(option_values)) != 4
            or answer not in option_map
            or normalize_answer_text(correct_option) != word_key
            or answer_is_visible
            or has_chinese_test_content
            or has_invalid_option
            or has_definition_question_style(question_text)
            or not word_key
            or not is_english_word_or_phrase(word)
            or (uses_phrase and phrase_count >= max_phrase_count)
            or question_key in seen_questions
            or option_set_key in seen_option_sets
            or word_key in seen_words
        ):
            continue

        seen_questions.add(question_key)
        seen_option_sets.add(option_set_key)
        seen_words.add(word_key)
        if uses_phrase:
            phrase_count += 1

        normalized.append(
            {
                "question": question_text,
                "options": option_map,
                "answer": answer,
                "explanation": to_traditional_chinese(
                    str(item.get("explanation", "")).strip()
                ),
                "word": word,
                "source": str(item.get("source", word)).strip() or word,
            }
        )

    if not normalized:
        raise ValueError("模型回傳的測驗題格式不完整，無法顯示互動測驗。")

    if len(normalized) != expected_count:
        raise ValueError(
            f"模型只回傳 {len(normalized)} 題有效、不重複且格式正確的測驗題，未達要求的 {expected_count} 題。"
        )

    return normalized


def extract_field_from_document(document: str, field_name: str) -> str:
    match = re.search(
        rf"^{re.escape(field_name)}:\s*(.*?)(?=\n[A-Z][A-Za-z ]+:\s|\Z)",
        document,
        flags=re.MULTILINE | re.DOTALL,
    )
    if not match:
        return ""
    value = re.sub(r"\s+", " ", match.group(1)).strip()
    return "" if value == "N/A" else value


def fallback_vocab_candidates(retrieved_results: List[Dict]) -> List[Dict]:
    candidates = []
    seen = set()

    for item in retrieved_results:
        metadata = item.get("metadata", {})
        document = str(item.get("document", ""))
        word = clean_choice_text(metadata.get("word") or extract_field_from_document(document, "Word"))
        word_key = normalize_answer_text(word)

        if (
            not word_key
            or word_key in seen
            or not is_english_word_or_phrase(word)
        ):
            continue

        seen.add(word_key)
        candidates.append(
            {
                "word": word,
                "part_of_speech": metadata.get("part_of_speech", ""),
                "zh_meaning": extract_field_from_document(document, "Chinese meaning"),
                "en_definition": extract_field_from_document(document, "English definition"),
            }
        )

    return candidates


def fallback_question_text(candidate: Dict, index: int = 0) -> str:
    part_of_speech = str(candidate.get("part_of_speech", "")).lower()
    noun_template = noun_semantic_template(candidate)
    if noun_template:
        return noun_template

    word = normalize_answer_text(candidate.get("word", ""))
    meaning_blob = normalize_answer_text(
        " ".join(
            [
                str(candidate.get("zh_meaning", "")),
                str(candidate.get("en_definition", "")),
            ]
        )
    )

    specific_patterns = [
        (["invoice", "receipt", "payment", "cost", "bill", "發票", "收據", "付款"], "The accounting department cannot process the payment without the original ______."),
        (["itinerary", "travel", "trip", "schedule", "行程"], "Please check the travel ______ before booking the hotel rooms."),
        (["applicant", "application", "candidate", "job", "申請"], "The human resources team contacted each ______ after the job interview."),
        (["shipment", "delivery", "warehouse", "logistics", "貨", "運送"], "The warehouse supervisor reported a delay in the overseas ______."),
        (["complaint", "customer", "service", "客訴", "抱怨"], "The customer service department responded to the ______ within one business day."),
        (["policy", "regulation", "rule", "safety", "規定", "政策"], "All employees must follow the updated company ______."),
        (["contract", "agreement", "合約"], "The legal department reviewed the ______ before the client signed it."),
        (["budget", "finance", "expense", "spending", "預算"], "The finance team reduced the project ______ after reviewing expected expenses."),
        (["deadline", "due", "期限"], "The project manager reminded the team about the upcoming ______."),
        (["equipment", "machine", "device", "office", "設備"], "The office manager ordered new ______ for the conference room."),
        (["reschedule", "postpone", "meeting", "delay", "延期", "改期"], "The manager had to ______ the meeting because several clients were unavailable."),
        (["cancel", "cancellation", "取消"], "The airline decided to ______ the flight because of severe weather."),
        (["arrange", "organize", "plan", "安排"], "The assistant will ______ transportation for the visiting executives."),
        (["approve", "approval", "accept", "核准"], "The supervisor must ______ the expense report before payment is issued."),
        (["confirm", "verify", "make sure", "確認"], "Please ______ the delivery address before sending the package."),
        (["recruit", "hire", "聘", "招募"], "The company plans to ______ additional staff for the new branch."),
    ]

    for keywords, template in specific_patterns:
        if any(keyword in word or keyword in meaning_blob for keyword in keywords):
            return template

    noun_templates = [
        "The accounting department reviewed the ______ before processing the payment.",
        "Please send the ______ to the client by Friday.",
        "The manager added the ______ to the meeting agenda.",
        "The customer service team attached the ______ to the response email.",
        "The company updated the ______ after the annual review.",
    ]
    verb_templates = [
        "The supervisor asked the assistant to ______ the client meeting for next week.",
        "Please ______ the shipping details before the order leaves the warehouse.",
        "The finance manager will ______ the reimbursement request after checking the receipts.",
        "The company needs to ______ the customer complaint before the end of the day.",
    ]
    adjective_templates = [
        "The client expects a ______ response from the customer service department.",
        "The revised shipping process is more ______ than the previous system.",
        "The sales report must be ______ before it is sent to senior management.",
        "The company needs a ______ procedure for handling travel expenses.",
    ]
    adverb_templates = [
        "The accounting team processed the invoice ______ after receiving approval.",
        "The assistant handled the customer complaint ______ during the call.",
        "The legal department reviewed the contract ______ before signing it.",
        "The warehouse staff updated the shipment record ______ after delivery.",
    ]
    phrase_templates = [
        "All employees must ______ company policies while working in the warehouse.",
        "The project manager asked the team to ______ the delay before calling the client.",
        "The company needs to ______ the increase in shipping costs this quarter.",
        "Please ______ the updated safety procedure before entering the warehouse.",
    ]

    if is_phrase(candidate.get("word", "")):
        templates = phrase_templates
    elif "noun" in part_of_speech or part_of_speech.startswith("n"):
        templates = noun_templates
    elif "adverb" in part_of_speech or part_of_speech.startswith("adv"):
        templates = adverb_templates
    elif "adjective" in part_of_speech or part_of_speech.startswith("adj"):
        templates = adjective_templates
    elif "verb" in part_of_speech or part_of_speech.startswith("v"):
        templates = verb_templates
    else:
        templates = [
            "The manager chose the best option to ______ the client's request.",
            "The team discussed the ______ during the project meeting.",
            "Please review the ______ before the department meeting.",
            "The company will ______ the new policy next month.",
        ]

    return templates[index % len(templates)]


def fallback_explanation(candidate: Dict) -> str:
    details = []
    if candidate.get("part_of_speech"):
        details.append(f"詞性：{candidate['part_of_speech']}")
    if candidate.get("zh_meaning"):
        details.append(f"中譯：{candidate['zh_meaning']}")
    if candidate.get("en_definition"):
        details.append(f"英文解釋：{candidate['en_definition']}")
    return to_traditional_chinese("；".join(details) or "此題由檢索到的單字資料自動產生。")


def rotate_items(items: List[str], offset: int) -> List[str]:
    if not items:
        return []
    offset = offset % len(items)
    return items[offset:] + items[:offset]


def unique_option_set(options: List[str]) -> Tuple[str, ...]:
    return tuple(sorted(normalize_answer_text(option) for option in options))


def part_of_speech_group(candidate: Dict) -> str:
    word = candidate.get("word", "")
    part_of_speech = str(candidate.get("part_of_speech", "")).lower()

    if is_phrase(word):
        return "phrase"
    if "noun" in part_of_speech or part_of_speech.startswith("n"):
        return "noun"
    if "verb" in part_of_speech or part_of_speech.startswith("v"):
        return "verb"
    if "adjective" in part_of_speech or part_of_speech.startswith("adj"):
        return "adjective"
    if "adverb" in part_of_speech or part_of_speech.startswith("adv"):
        return "adverb"
    return "other"


def build_quiz_blueprints(
    candidates: List[Dict],
    expected_count: int,
    used_words: Union[set, None] = None,
    used_option_sets: Union[set, None] = None,
) -> List[Dict]:
    used = set(used_words or set())
    used_sets = set(used_option_sets or set())
    selected = []
    phrase_limit = max(1, expected_count // 4)
    phrase_count = 0

    for candidate in candidates:
        word_key = normalize_answer_text(candidate["word"])
        uses_phrase = is_phrase(candidate["word"])

        if word_key in used or (uses_phrase and phrase_count >= phrase_limit):
            continue

        selected.append(candidate)
        used.add(word_key)
        if uses_phrase:
            phrase_count += 1

        if len(selected) == expected_count:
            break

    if len(selected) < expected_count:
        raise ValueError("檢索到的單字不足，無法產生足夠的測驗題。")

    blueprints = []
    answer_letters = ["A", "B", "C", "D"]

    for index, candidate in enumerate(selected):
        correct_word = candidate["word"]
        options = [correct_word]
        correct_key = normalize_answer_text(correct_word)
        candidates_by_key = {
            normalize_answer_text(other["word"]): other
            for other in candidates
        }
        target_group = part_of_speech_group(candidate)
        same_pos_distractors = [
            other["word"]
            for other in candidates
            if (
                normalize_answer_text(other["word"]) != correct_key
                and part_of_speech_group(other) == target_group
            )
        ]
        fallback_distractors = [
            other["word"]
            for other in candidates
            if (
                normalize_answer_text(other["word"]) != correct_key
                and part_of_speech_group(other) != target_group
            )
        ]
        distractor_pool = rotate_items(same_pos_distractors, index * 3) + rotate_items(
            fallback_distractors,
            index * 5,
        )

        for distractor in distractor_pool:
            if normalize_answer_text(distractor) not in {
                normalize_answer_text(option) for option in options
            }:
                options.append(distractor)
            if len(options) == 4:
                break

        if len(options) < 4:
            raise ValueError("檢索到的可用干擾選項不足，無法產生測驗題。")

        if unique_option_set(options) in used_sets:
            for offset in range(1, len(distractor_pool)):
                alternate_options = [correct_word]
                for distractor in rotate_items(distractor_pool, offset):
                    if normalize_answer_text(distractor) not in {
                        normalize_answer_text(option) for option in alternate_options
                    }:
                        alternate_options.append(distractor)
                    if len(alternate_options) == 4:
                        break

                if len(alternate_options) == 4 and unique_option_set(alternate_options) not in used_sets:
                    options = alternate_options
                    break

        used_sets.add(unique_option_set(options))

        rotation = index % 4
        options = options[rotation:] + options[:rotation]
        option_map = dict(zip(answer_letters, options))
        answer = next(
            letter
            for letter, option in option_map.items()
            if normalize_answer_text(option) == normalize_answer_text(correct_word)
        )

        blueprints.append(
            {
                "target": candidate,
                "option_candidates": [
                    candidates_by_key.get(normalize_answer_text(option), {"word": option})
                    for option in option_map.values()
                ],
                "word": correct_word,
                "options": option_map,
                "answer": answer,
                "source": correct_word,
            }
        )

    return blueprints


def build_questions_from_blueprints(blueprints: List[Dict]) -> List[Dict]:
    fallback_questions = []

    for index, blueprint in enumerate(blueprints):
        candidate = blueprint["target"]
        fallback_questions.append(
            {
                "question": fallback_question_text(candidate, index),
                "options": blueprint["options"],
                "answer": blueprint["answer"],
                "explanation": fallback_explanation(candidate),
                "word": blueprint["word"],
                "source": blueprint["source"],
            }
        )

    return fallback_questions


def build_fallback_quiz_questions(
    retrieved_results: List[Dict],
    expected_count: int,
    used_words: Union[set, None] = None,
    used_option_sets: Union[set, None] = None,
) -> List[Dict]:
    candidates = fallback_vocab_candidates(retrieved_results)
    blueprints = build_quiz_blueprints(
        candidates,
        expected_count,
        used_words=used_words,
        used_option_sets=used_option_sets,
    )
    return build_questions_from_blueprints(blueprints)


def format_quiz_blueprints(blueprints: List[Dict]) -> str:
    lines = []
    for index, blueprint in enumerate(blueprints, start=1):
        option_text = "\n".join(
            f"  {letter}. {text}"
            for letter, text in blueprint["options"].items()
        )
        target = blueprint["target"]
        lines.append(
            f"""Question {index} fixed blueprint:
Target word: {blueprint['word']}
Part of speech: {target.get('part_of_speech', '')}
Chinese meaning: {target.get('zh_meaning', '')}
English definition: {target.get('en_definition', '')}
Answer: {blueprint['answer']}
Options:
{option_text}"""
        )
    return "\n\n".join(lines)


def format_blueprint_context(blueprints: List[Dict]) -> str:
    lines = []
    seen_words = set()

    for blueprint in blueprints:
        option_words = list(blueprint["options"].values())
        for word in [blueprint["word"], *option_words]:
            word_key = normalize_answer_text(word)
            if word_key in seen_words:
                continue
            seen_words.add(word_key)

            candidate = None
            for possible_candidate in blueprint.get("option_candidates", [blueprint["target"]]):
                if normalize_answer_text(possible_candidate.get("word", "")) == word_key:
                    candidate = possible_candidate
                    break

            if candidate:
                lines.append(
                    "\n".join(
                        [
                            f"Word: {candidate.get('word', word)}",
                            f"Part of speech: {candidate.get('part_of_speech', '')}",
                            f"Chinese meaning: {candidate.get('zh_meaning', '')}",
                            f"English definition: {candidate.get('en_definition', '')}",
                        ]
                    )
                )
            else:
                lines.append(f"Word: {word}")

    return "\n\n".join(lines)


def apply_fixed_blueprints(questions: List[Dict], blueprints: List[Dict]) -> List[Dict]:
    if len(questions) != len(blueprints):
        raise ValueError("模型回傳題數與固定選項規格不一致。")

    fixed_questions = []
    seen_question_texts = set()

    for index, (question, blueprint) in enumerate(zip(questions, blueprints)):
        expected_word = normalize_answer_text(blueprint["word"])
        returned_word = normalize_answer_text(question.get("word", ""))
        returned_correct = normalize_answer_text(
            question.get("options", {}).get(question.get("answer", ""), "")
        )

        if returned_word not in {"", expected_word} and returned_correct != expected_word:
            raise ValueError(
                f"第 {index + 1} 題 target word 與固定規格不一致。"
            )

        question_text = normalize_blank_marker(question.get("question", ""))
        question_key = re.sub(r"\s+", " ", question_text.lower())
        if question_key in seen_question_texts:
            raise ValueError(f"第 {index + 1} 題題幹與前題重複。")
        seen_question_texts.add(question_key)

        fixed_questions.append(
            {
                "question": question_text,
                "options": blueprint["options"],
                "answer": blueprint["answer"],
                "explanation": question.get("explanation", ""),
                "word": blueprint["word"],
                "source": blueprint["source"],
            }
        )

    return fixed_questions


def extract_vocab_candidates(retrieved_results: List[Dict]) -> List[Dict]:
    """
    Convert retrieved Chroma results into unique vocabulary candidates.
    Each candidate contains word, part_of_speech, document, metadata, and distance.
    """
    candidates = []
    seen_words = set()

    for item in retrieved_results:
        metadata = item.get("metadata", {}) or {}
        word = str(metadata.get("word", "")).strip()
        pos = str(metadata.get("part_of_speech", "")).strip()
        document = item.get("document", "")

        if not word:
            continue

        word_key = word.lower()
        if word_key in seen_words:
            continue

        seen_words.add(word_key)

        candidates.append(
            {
                "word": word,
                "part_of_speech": pos,
                "document": document,
                "metadata": metadata,
                "distance": item.get("distance"),
            }
        )

    return candidates


def normalize_quiz_pos(pos: str) -> str:
    """
    Normalize part of speech for distractor matching.
    If a word has multiple POS values like noun/verb, use the first one.
    """
    if not pos:
        return ""

    pos = str(pos).strip().lower()

    if "/" in pos:
        pos = pos.split("/")[0].strip()

    return pos


NOUN_SEMANTIC_CATEGORIES = [
    {
        "name": "payment_document",
        "keywords": [
            "invoice",
            "bill",
            "receipt",
            "estimate",
            "quotation",
            "quote",
            "statement",
            "發票",
            "帳單",
            "收據",
            "估價",
            "報價",
        ],
        "templates": {
            "invoice": "The accounting department reviewed the ______ before processing the payment.",
            "bill": "The accounting department reviewed the ______ before processing the payment.",
            "receipt": "Employees must submit a ______ to receive reimbursement for travel expenses.",
            "estimate": "The client requested an ______ before approving the repair work.",
            "quotation": "The client requested a ______ before approving the repair work.",
            "quote": "The client requested a ______ before approving the repair work.",
            "statement": "The finance department checked the monthly ______ before closing the account.",
            "default": "The accounting department reviewed the ______ before processing the payment.",
        },
    },
    {
        "name": "business_document",
        "keywords": [
            "contract",
            "agreement",
            "proposal",
            "report",
            "form",
            "application",
            "合約",
            "協議",
            "提案",
            "報告",
            "表格",
            "申請",
        ],
        "templates": {
            "contract": "The legal department reviewed the ______ before the client signed it.",
            "agreement": "The legal department reviewed the ______ before the client signed it.",
            "proposal": "The sales team sent the ______ to the client after the meeting.",
            "report": "The accounting department reviewed the ______ before the monthly meeting.",
            "form": "Please complete the ______ before submitting the job application.",
            "application": "The human resources team reviewed the ______ before scheduling an interview.",
            "default": "The legal department reviewed the ______ before the client signed it.",
        },
    },
    {
        "name": "travel_schedule",
        "keywords": [
            "itinerary",
            "reservation",
            "booking",
            "ticket",
            "fare",
            "schedule",
            "timetable",
            "行程",
            "訂位",
            "票",
            "票價",
            "時程",
            "時間表",
        ],
        "templates": {
            "itinerary": "The travel coordinator updated the ______ before booking the hotel rooms.",
            "reservation": "The assistant confirmed the ______ before the executive arrived at the hotel.",
            "booking": "The assistant confirmed the ______ before the executive arrived at the hotel.",
            "ticket": "The travel agent sent the electronic ______ to the employee before the trip.",
            "fare": "The travel coordinator compared the ______ before booking the flight.",
            "schedule": "The manager added the updated ______ to the meeting agenda.",
            "timetable": "The manager added the updated ______ to the meeting agenda.",
            "default": "The travel coordinator updated the ______ before booking the hotel rooms.",
        },
    },
    {
        "name": "people_role",
        "keywords": [
            "applicant",
            "candidate",
            "employee",
            "manager",
            "supervisor",
            "client",
            "customer",
            "staff",
            "personnel",
            "申請者",
            "候選",
            "員工",
            "經理",
            "主管",
            "客戶",
        ],
        "templates": {
            "applicant": "The hiring manager contacted each ______ after the job interview.",
            "candidate": "The hiring manager contacted each ______ after the job interview.",
            "employee": "Every ______ must complete the safety training by Friday.",
            "manager": "The ______ approved the revised project schedule this morning.",
            "supervisor": "The ______ approved the revised project schedule this morning.",
            "client": "The sales team sent the proposal to the ______ after the meeting.",
            "customer": "The service representative called the ______ to discuss the complaint.",
            "staff": "All ______ must complete the safety training by Friday.",
            "personnel": "All ______ must complete the safety training by Friday.",
            "default": "The hiring manager contacted each ______ after the job interview.",
        },
    },
    {
        "name": "logistics_inventory",
        "keywords": [
            "shipment",
            "delivery",
            "inventory",
            "warehouse",
            "package",
            "order",
            "cargo",
            "貨",
            "運送",
            "交貨",
            "庫存",
            "倉庫",
            "包裹",
            "訂單",
        ],
        "templates": {
            "shipment": "The warehouse supervisor reported a delay in the overseas ______.",
            "delivery": "The customer service team apologized for the delayed ______.",
            "inventory": "The warehouse team checked the ______ before placing a new order.",
            "warehouse": "The logistics manager inspected the ______ before the shipment arrived.",
            "package": "The receptionist signed for the ______ when it arrived at the office.",
            "order": "The sales department confirmed the ______ before sending it to the warehouse.",
            "cargo": "The logistics team inspected the ______ before it left the port.",
            "default": "The warehouse supervisor reported a delay in the overseas ______.",
        },
    },
    {
        "name": "policy_procedure",
        "keywords": [
            "policy",
            "regulation",
            "rule",
            "procedure",
            "guideline",
            "requirement",
            "instruction",
            "政策",
            "規定",
            "規則",
            "程序",
            "方針",
            "要求",
        ],
        "templates": {
            "policy": "The company updated the ______ after the annual review.",
            "regulation": "All employees must follow the safety ______ in the warehouse.",
            "rule": "All employees must follow the safety ______ in the warehouse.",
            "procedure": "The company updated the ______ after the annual review.",
            "guideline": "The company updated the ______ after the annual review.",
            "requirement": "The job posting lists each ______ for the position.",
            "instruction": "The supervisor gave the ______ before the equipment inspection.",
            "default": "The company updated the ______ after the annual review.",
        },
    },
    {
        "name": "finance_resource",
        "keywords": [
            "budget",
            "fund",
            "expense",
            "revenue",
            "profit",
            "cost",
            "account",
            "預算",
            "資金",
            "費用",
            "收入",
            "利潤",
            "成本",
            "帳戶",
        ],
        "templates": {
            "budget": "The finance team revised the project ______ after the annual review.",
            "fund": "The committee created a special ______ for employee training.",
            "expense": "The manager recorded each travel ______ in the monthly report.",
            "revenue": "The sales report showed an increase in quarterly ______.",
            "profit": "The financial report showed a higher ______ than expected.",
            "cost": "The manager compared the ______ before choosing a supplier.",
            "account": "The bank opened a new ______ for the company's payroll.",
            "default": "The finance team revised the project ______ after the annual review.",
        },
    },
    {
        "name": "office_equipment",
        "keywords": [
            "equipment",
            "device",
            "machine",
            "computer",
            "printer",
            "desk",
            "furniture",
            "設備",
            "裝置",
            "機器",
            "電腦",
            "印表機",
            "桌",
            "家具",
        ],
        "templates": {
            "equipment": "The office manager ordered new ______ for the conference room.",
            "device": "The technician tested each ______ before the training session.",
            "machine": "The technician repaired the ______ before the office reopened.",
            "computer": "The IT department replaced the ______ after the system failure.",
            "printer": "The assistant loaded paper into the ______ before printing the reports.",
            "desk": "The office manager ordered a new ______ for the reception area.",
            "furniture": "The office manager ordered new ______ for the conference room.",
            "default": "The office manager ordered new ______ for the conference room.",
        },
    },
    {
        "name": "event_location",
        "keywords": [
            "venue",
            "location",
            "conference",
            "session",
            "assembly",
            "seminar",
            "workshop",
            "場地",
            "地點",
            "會議",
            "研討會",
        ],
        "templates": {
            "venue": "The event coordinator confirmed the ______ before sending the invitations.",
            "location": "The event coordinator confirmed the ______ before sending the invitations.",
            "conference": "The marketing team attended the annual ______ in Taipei.",
            "session": "The training ______ will begin at nine o'clock tomorrow morning.",
            "assembly": "The annual ______ will take place in the main auditorium.",
            "seminar": "The company hosted a ______ on workplace safety.",
            "workshop": "The company hosted a ______ on customer service skills.",
            "default": "The event coordinator confirmed the ______ before sending the invitations.",
        },
    },
]


MVP_BANNED_TARGET_WORDS = {
    "capex",
    "business capital",
    "gathering",
    "facility",
    "banknote",
    "assembly",
    "establishment",
    "encounter",
    "junction",
    "acronym",
}


TEMPLATE_ALLOWED_WORDS = {
    "The accounting department reviewed the ______ before processing the payment.": {
        "invoice",
        "receipt",
        "bill",
        "statement",
        "expense report",
        "sales slip",
    },
    "Employees must submit a ______ to receive reimbursement for travel expenses.": {
        "receipt",
        "expense report",
    },
    "The client requested an ______ before approving the repair work.": {
        "estimate",
    },
    "The client requested a ______ before approving the repair work.": {
        "quotation",
        "quote",
    },
    "The finance department checked the monthly ______ before closing the account.": {
        "statement",
    },
    "The legal department reviewed the ______ before the client signed it.": {
        "contract",
        "agreement",
    },
    "The sales team sent the ______ to the client after the meeting.": {
        "proposal",
    },
    "The accounting department reviewed the ______ before the monthly meeting.": {
        "report",
        "sales report",
        "financial report",
    },
    "Please complete the ______ before submitting the job application.": {
        "form",
        "application",
    },
    "The human resources team reviewed the ______ before scheduling an interview.": {
        "application",
    },
    "The travel coordinator updated the ______ before booking the hotel rooms.": {
        "itinerary",
        "schedule",
    },
    "The assistant confirmed the ______ before the executive arrived at the hotel.": {
        "reservation",
        "booking",
    },
    "The travel agent sent the electronic ______ to the employee before the trip.": {
        "ticket",
    },
    "The travel coordinator compared the ______ before booking the flight.": {
        "fare",
    },
    "The manager added the updated ______ to the meeting agenda.": {
        "schedule",
        "timetable",
    },
    "The hiring manager contacted each ______ after the job interview.": {
        "applicant",
        "candidate",
    },
    "Every ______ must complete the safety training by Friday.": {
        "employee",
    },
    "The ______ approved the revised project schedule this morning.": {
        "manager",
        "supervisor",
    },
    "The service representative called the ______ to discuss the complaint.": {
        "customer",
        "client",
    },
    "The sales team sent the proposal to the ______ after the meeting.": {
        "client",
    },
    "All ______ must complete the safety training by Friday.": {
        "staff",
        "personnel",
    },
    "The warehouse supervisor reported a delay in the overseas ______.": {
        "shipment",
        "delivery",
    },
    "The customer service team apologized for the delayed ______.": {
        "delivery",
        "shipment",
    },
    "The warehouse team checked the ______ before placing a new order.": {
        "inventory",
        "stock",
    },
    "The logistics manager inspected the ______ before the shipment arrived.": {
        "warehouse",
    },
    "The receptionist signed for the ______ when it arrived at the office.": {
        "package",
        "parcel",
    },
    "The sales department confirmed the ______ before sending it to the warehouse.": {
        "order",
    },
    "The logistics team inspected the ______ before it left the port.": {
        "cargo",
    },
    "The company updated the ______ after the annual review.": {
        "policy",
        "procedure",
        "guideline",
    },
    "All employees must follow the safety ______ in the warehouse.": {
        "regulation",
        "rule",
        "procedure",
    },
    "The job posting lists each ______ for the position.": {
        "requirement",
    },
    "The supervisor gave the ______ before the equipment inspection.": {
        "instruction",
    },
    "The finance team revised the project ______ after the annual review.": {
        "budget",
    },
    "The committee created a special ______ for employee training.": {
        "fund",
    },
    "The manager recorded each travel ______ in the monthly report.": {
        "expense",
    },
    "The sales report showed an increase in quarterly ______.": {
        "revenue",
    },
    "The financial report showed a higher ______ than expected.": {
        "profit",
    },
    "The manager compared the ______ before choosing a supplier.": {
        "cost",
    },
    "The bank opened a new ______ for the company's payroll.": {
        "account",
    },
    "The office manager ordered new ______ for the conference room.": {
        "equipment",
        "furniture",
    },
    "The technician tested each ______ before the training session.": {
        "device",
    },
    "The technician repaired the ______ before the office reopened.": {
        "machine",
    },
    "The IT department replaced the ______ after the system failure.": {
        "computer",
    },
    "The assistant loaded paper into the ______ before printing the reports.": {
        "printer",
    },
    "The office manager ordered a new ______ for the reception area.": {
        "desk",
    },
    "The event coordinator confirmed the ______ before sending the invitations.": {
        "venue",
        "location",
    },
    "The marketing team attended the annual ______ in Taipei.": {
        "conference",
    },
    "The training ______ will begin at nine o'clock tomorrow morning.": {
        "session",
    },
    "The company hosted a ______ on workplace safety.": {
        "seminar",
    },
    "The company hosted a ______ on customer service skills.": {
        "workshop",
    },
}


def noun_semantic_profile(candidate: Dict) -> Union[Dict, None]:
    word = normalize_answer_text(candidate.get("word", ""))
    document = str(candidate.get("document", ""))
    meaning_blob = normalize_answer_text(
        " ".join(
            [
                word,
                extract_field_from_document(document, "Chinese meaning"),
                extract_field_from_document(document, "English definition"),
            ]
        )
    )

    for category in NOUN_SEMANTIC_CATEGORIES:
        if any(semantic_keyword_matches(keyword, meaning_blob) for keyword in category["keywords"]):
            templates = category["templates"]
            template = templates.get(word, templates["default"])
            if not is_allowed_template_target(template, word):
                return None
            return {
                "category": category["name"],
                "template": template,
            }

    return None


def is_allowed_template_target(template: str, word: str) -> bool:
    word_key = normalize_answer_text(word)
    if word_key in MVP_BANNED_TARGET_WORDS:
        return False

    allowed_words = TEMPLATE_ALLOWED_WORDS.get(template)
    if not allowed_words:
        return False

    return word_key in allowed_words


def semantic_keyword_matches(keyword: str, text: str) -> bool:
    keyword = normalize_answer_text(keyword)
    if contains_cjk(keyword):
        return keyword in text
    return bool(re.search(rf"\b{re.escape(keyword)}\b", text))


def noun_semantic_template(candidate: Dict) -> str:
    profile = noun_semantic_profile(candidate)
    if profile:
        return profile["template"]

    return ""


def is_noun_quiz_candidate(candidate: Dict) -> bool:
    return normalize_quiz_pos(candidate.get("part_of_speech", "")) == "noun"


def build_quiz_plan(
    candidates: List[Dict],
    num_questions: int,
    used_words: List[str] = None,
    seed: int = None,
) -> List[Dict]:
    """
    Build a fixed quiz plan before calling the LLM.

    Requirements:
    - unique target word per question
    - 4 unique options inside each question
    - avoid repeating option sets across questions
    - avoid reusing option words across questions when possible
    - avoid previously used words from earlier quiz generations when possible
    - prefer distractors with the same part of speech as the target
    """
    if used_words is None:
        used_words = []

    used_words_set = set(w.lower() for w in used_words if w)

    if seed is None:
        seed = int(time.time())

    rng = random.Random(seed)

    unique_candidates = []
    seen = set()

    for c in candidates:
        word = str(c.get("word", "")).strip()
        word_key = word.lower()

        if not word or word_key in seen:
            continue

        seen.add(word_key)
        unique_candidates.append(c)

    noun_candidates = [
        c for c in unique_candidates
        if is_noun_quiz_candidate(c)
    ]
    suitable_targets = [
        c for c in noun_candidates
        if noun_semantic_profile(c)
    ]

    fresh_candidates = [
        c for c in unique_candidates
        if c["word"].lower() not in used_words_set
        and is_noun_quiz_candidate(c)
        and noun_semantic_profile(c)
    ]

    if len(fresh_candidates) >= num_questions * 4:
        candidate_pool = fresh_candidates
    else:
        candidate_pool = suitable_targets

    if len(candidate_pool) < 4:
        raise ValueError("高品質候選題不足，請降低題數、提高 top_k，或更換較廣的主題。")

    rng.shuffle(candidate_pool)

    quiz_plan = []
    used_targets_in_this_quiz = set()
    used_options_in_this_quiz = set()
    used_option_sets = set()
    used_templates_in_this_quiz = set()
    candidates_by_category = {}

    for candidate in candidate_pool:
        profile = noun_semantic_profile(candidate)
        if not profile:
            continue
        candidates_by_category.setdefault(profile["category"], []).append(candidate)

    category_names = list(candidates_by_category.keys())
    rng.shuffle(category_names)

    for planning_pass in range(3):
        if len(quiz_plan) >= num_questions:
            break

        made_progress = False

        for category_name in category_names:
            if len(quiz_plan) >= num_questions:
                break

            category_candidates = candidates_by_category.get(category_name, [])
            available_targets = [
                c for c in category_candidates
                if c["word"].lower() not in used_targets_in_this_quiz
                and (
                    planning_pass > 0
                    or c["word"].lower() not in used_options_in_this_quiz
                )
            ]

            if not available_targets:
                continue

            target = rng.choice(available_targets)
            target_key = target["word"].lower()
            target_word = target["word"]
            target_profile = noun_semantic_profile(target)
            target_template = target_profile["template"]
            target_category = target_profile["category"]
            if target_template in used_templates_in_this_quiz:
                continue
            unused_other_category = [
                c for c in candidate_pool
                if c["word"].lower() != target_key
                and c["word"].lower() not in used_options_in_this_quiz
                and noun_semantic_profile(c)
                and noun_semantic_profile(c)["category"] != target_category
            ]
            reusable_other_category = [
                c for c in candidate_pool
                if c["word"].lower() != target_key
                and noun_semantic_profile(c)
                and noun_semantic_profile(c)["category"] != target_category
            ]
            unused_same_category = [
                c for c in category_candidates
                if c["word"].lower() != target_key
                and c["word"].lower() not in used_options_in_this_quiz
            ]
            reusable_same_category = [
                c for c in category_candidates
                if c["word"].lower() != target_key
            ]

            if len(unused_other_category) >= 3:
                distractor_pool = unused_other_category
            elif len(reusable_other_category) >= 3:
                distractor_pool = reusable_other_category
            elif len(unused_same_category) >= 3:
                distractor_pool = unused_same_category
            elif len(reusable_same_category) >= 3:
                distractor_pool = reusable_same_category
            else:
                continue

            distractors = rng.sample(distractor_pool, 3)
            options = [target] + distractors
            rng.shuffle(options)

            option_words = [o["word"] for o in options]
            option_keys = [w.lower() for w in option_words]

            if len(set(option_keys)) < 4:
                continue

            option_set_key = tuple(sorted(option_keys))

            if option_set_key in used_option_sets and planning_pass < 2:
                continue

            answer_index = option_words.index(target_word)
            answer_letter = ["A", "B", "C", "D"][answer_index]

            quiz_plan.append(
                {
                    "target_word": target_word,
                    "target_part_of_speech": target.get("part_of_speech", ""),
                    "answer": answer_letter,
                    "options": {
                        "A": option_words[0],
                        "B": option_words[1],
                        "C": option_words[2],
                        "D": option_words[3],
                    },
                    "target_context": target["document"],
                    "option_contexts": [o["document"] for o in options],
                    "semantic_template": target_template,
                    "semantic_category": target_category,
                }
            )

            used_targets_in_this_quiz.add(target_key)
            used_option_sets.add(option_set_key)
            used_options_in_this_quiz.update(option_keys)
            used_templates_in_this_quiz.add(target_template)
            made_progress = True

        if not made_progress:
            continue

    if len(quiz_plan) < num_questions:
        raise ValueError("高品質候選題不足，請降低題數、提高 top_k，或更換較廣的主題。")

    return quiz_plan


def format_quiz_plan(quiz_plan: List[Dict]) -> str:
    """
    Format the fixed quiz plan for the LLM.
    The LLM must use the exact target word, answer letter, and options.
    """
    parts = []

    for i, item in enumerate(quiz_plan, start=1):
        options = item["options"]
        target_context = item.get("target_context", "")
        target_meaning = extract_field_from_document(target_context, "Chinese meaning")
        target_definition = extract_field_from_document(target_context, "English definition")

        parts.append(
            f"""Question {i} plan:
Target word: {item["target_word"]}
Target part of speech: {item["target_part_of_speech"]}
Target meaning: {target_meaning}
Target definition: {target_definition}
Semantic category: {item.get("semantic_category", "")}
Recommended noun-slot template: {item.get("semantic_template", "")}
Collocation instruction: Use this semantic category and template as the context. Distractors may come from other semantic categories on purpose; write the sentence so only the target word is semantically natural.
Correct answer: {item["answer"]}

Fixed options:
(A) {options["A"]}
(B) {options["B"]}
(C) {options["C"]}
(D) {options["D"]}

Target context:
{item["target_context"]}

Option contexts:
{chr(10).join(item["option_contexts"])}
"""
        )

    return "\n\n".join(parts)


def get_words_from_quiz_plan(quiz_plan: List[Dict]) -> List[str]:
    """
    Return all words used in this quiz plan, including targets and all options.
    Streamlit will store these words in session_state to avoid repeating them
    in the next quiz generation.
    """
    words = []

    for item in quiz_plan:
        words.append(item["target_word"])
        words.extend(list(item["options"].values()))

    seen = set()
    unique_words = []

    for word in words:
        key = word.lower()
        if key not in seen:
            seen.add(key)
            unique_words.append(word)

    return unique_words


def validate_option_diversity(quiz_plan: List[Dict]) -> List[str]:
    """
    Validate diversity of options before LLM generation.
    """
    warnings = []
    seen_option_sets = set()
    all_option_words = []

    for i, item in enumerate(quiz_plan, start=1):
        option_words = list(item["options"].values())
        option_keys = [w.lower() for w in option_words]

        if len(set(option_keys)) < 4:
            warnings.append(f"第 {i} 題有重複選項。")

        option_set = tuple(sorted(option_keys))
        if option_set in seen_option_sets:
            warnings.append(f"第 {i} 題重複使用前面出現過的選項組合。")

        seen_option_sets.add(option_set)
        all_option_words.extend(option_keys)

    repeated_words = sorted(
        word for word in set(all_option_words)
        if all_option_words.count(word) > 1
    )

    if repeated_words:
        warnings.append(
            "以下選項單字在不同題目中重複出現："
            + ", ".join(repeated_words[:20])
        )

    return warnings


def build_question_from_plan_item(
    item: Dict,
    index: int,
    question_text: str = "",
    explanation: str = "",
) -> Dict:
    target_word = item["target_word"]
    target_context = item.get("target_context", "")
    candidate = {
        "word": target_word,
        "part_of_speech": item.get("target_part_of_speech", ""),
        "document": target_context,
        "zh_meaning": extract_field_from_document(target_context, "Chinese meaning"),
        "en_definition": extract_field_from_document(target_context, "English definition"),
    }

    question_text = normalize_blank_marker(question_text)
    if (
        not question_text
        or question_text.count("______") != 1
        or has_definition_question_style(question_text)
    ):
        question_text = fallback_question_text(candidate, index)

    if not explanation:
        explanation = fallback_explanation(candidate)

    return {
        "question": question_text,
        "options": item["options"],
        "answer": item["answer"],
        "explanation": to_traditional_chinese(explanation.strip()),
        "word": target_word,
        "source": target_word,
    }


def parse_generated_quiz_text(quiz_text: str, quiz_plan: List[Dict]) -> List[Dict]:
    """
    Extract question sentences and explanations from LLM text, while keeping
    Python-generated options, answers, and target words fixed.
    """
    questions = []

    for index, item in enumerate(quiz_plan, start=1):
        block_match = re.search(
            rf"Question\s*{index}\s*:\s*(.*?)(?=\n\s*Question\s*{index + 1}\s*:|\Z)",
            quiz_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        block = block_match.group(1).strip() if block_match else ""

        question_lines = []
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line:
                if question_lines:
                    break
                continue
            if re.match(r"^\([A-Da-d]\)", line):
                break
            if re.match(r"^(Answer|Target word|Explanation|Source)\s*:", line, flags=re.IGNORECASE):
                break
            question_lines.append(line)

        question_text = " ".join(question_lines).strip()

        explanation = ""
        explanation_match = re.search(
            r"Explanation\s*:\s*(.*?)(?=\n\s*Source\s*:|\n\s*Question\s*\d+\s*:|\Z)",
            block,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if explanation_match:
            explanation = re.sub(r"\s+", " ", explanation_match.group(1)).strip()

        questions.append(
            build_question_from_plan_item(
                item,
                index - 1,
                question_text=question_text,
                explanation=explanation,
            )
        )

    return questions


def generate_quiz(
    topic: str,
    num_questions: int = 5,
    top_k: int = 60,
    used_words: List[str] = None,
):
    """
    Generate TOEIC Reading Part 5 style quiz.

    This version:
    - retrieves more candidate words
    - builds a fixed quiz plan in Python
    - gives the fixed plan to the LLM
    - avoids repeating options within the same quiz
    - avoids previously used words across quiz generations when possible
    """
    if used_words is None:
        used_words = []

    query = topic.strip() if topic.strip() else "TOEIC vocabulary business office finance meeting travel"

    retrieval_k = max(top_k, num_questions * 12)

    retrieved = retrieve_context(query, top_k=retrieval_k)
    candidates = extract_vocab_candidates(retrieved)

    seed = int(time.time())

    quiz_plan = build_quiz_plan(
        candidates=candidates,
        num_questions=num_questions,
        used_words=used_words,
        seed=seed,
    )

    quiz_plan_text = format_quiz_plan(quiz_plan)

    user_prompt = f"""Vocabulary context and fixed quiz plan:
{quiz_plan_text}

Quiz requirement:
請根據以上 fixed quiz plan 產生 {len(quiz_plan)} 題 TOEIC Reading Part 5 風格的句子填空題。

主題或需求：
{topic}

出題要求：
1. MVP 版本只產生名詞題，每題必須是一個自然的英文商務、職場、會議、財務、旅遊、物流、客服或辦公室情境句子。
2. 每題句子只能有一個空格，使用 ______ 表示。
3. 每題必須使用 fixed options 中指定的 A-D 選項。
4. 不可以更換、刪除、改寫或新增選項。
5. 不可以改變 fixed options 的 A-D 順序。
6. 每題的正確答案必須使用 fixed quiz plan 中指定的 Correct answer。
7. 每題的 Target word 必須等於 fixed quiz plan 中指定的 Target word。
8. 不可以出現「The word meaning...」或「The TOEIC vocabulary term meaning...」這種定義查詢題。
9. 題目應測驗語境中的單字使用，而不是直接問中文翻譯。
10. 請使用繁體中文提供解析。
11. 請明確標示 Answer、Target word、Explanation、Source。
12. 不要聲稱題目是 TOEIC 官方題目。
13. 題幹必須根據 fixed quiz plan 中的 Target meaning、Target definition、Target context 設計。
14. Python 已經替每題指定 Semantic category 與 Recommended noun-slot template；請使用該 category/template 作為語意情境，不要自行改成其他情境。
15. 空格前後必須和 Target word 形成自然搭配或自然語法結構。
16. 不要創造句子，除非 Target word 是空格中最自然、最精確的答案。
17. 每題空格必須是 noun slot，固定選項都應作為 noun 或 noun phrase 使用。
18. 避免這些泛用模板，除非 target word 明顯是唯一自然答案：
    - The manager reviewed the ______ before approving the request.
    - The team discussed the ______ during the meeting.
    - The sales department prepared the ______ for the client.
19. 不要使用動詞空格模板，例如 "Please ______ ..."。
20. 優先使用具體 TOEIC 情境：accounting department、payment processing、shipment delay、meeting schedule、job application、customer complaint、sales report、travel itinerary、office equipment、company policy。
21. 如果 Recommended noun-slot template 適合 target word，優先使用或只做小幅潤飾；如果不適合，改寫成同一 Semantic category 的具體 noun-slot 句子。
22. 在輸出前請自行檢查：target word 是否語意自然、blank 是否為 noun slot、是否只有一個 ______、A-D 是否完全未改、是否只有一個最佳答案。不要輸出自我檢查內容。

輸出格式：

Question 1:
[English sentence with one blank: ______]

(A) [fixed option A]
(B) [fixed option B]
(C) [fixed option C]
(D) [fixed option D]

Answer: [A/B/C/D]
Target word: [correct word]
Explanation: [繁體中文解析]
Source: [target word source]
"""

    try:
        quiz_text = call_ollama(QUIZ_GENERATION_PROMPT, user_prompt, timeout=90)
        quiz_questions = parse_generated_quiz_text(quiz_text, quiz_plan)
    except requests.RequestException as e:
        quiz_text = ""
        quiz_questions = [
            build_question_from_plan_item(item, index)
            for index, item in enumerate(quiz_plan)
        ]
        style_warnings = [
            f"模型生成失敗，已改用固定題型產生題目：{e}"
        ]
    else:
        style_warnings = validate_quiz_style(quiz_text)

    if len(quiz_questions) != num_questions:
        quiz_questions = [
            build_question_from_plan_item(item, index)
            for index, item in enumerate(quiz_plan)
        ]
        style_warnings.append("模型輸出的題數不符合需求，已改用固定題型產生題目。")

    style_warnings.extend(validate_quiz_questions_style(quiz_questions))
    diversity_warnings = validate_option_diversity(quiz_plan)
    warnings = diversity_warnings + style_warnings

    used_words_this_round = get_words_from_quiz_plan(quiz_plan)

    return quiz_questions, retrieved, warnings, used_words_this_round
