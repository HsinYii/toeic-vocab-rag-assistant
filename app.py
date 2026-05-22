import os
import re

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from rag_core import (
    build_vector_db,
    generate_answer,
    generate_quiz,
    load_local_vocab_files,
    merge_vocab_frames,
    parse_vocab_text,
    read_uploaded_text,
    retrieve_context,
    to_traditional_chinese,
)

load_dotenv()

st.set_page_config(
    page_title="多益單字 RAG 練習系統",
    page_icon="📘",
    layout="wide",
)

st.title("多益單字 RAG 練習系統")
st.caption("建立自己的單字庫，練習多益單字問答與句子填空。")

if "vocab_df" not in st.session_state:
    st.session_state["vocab_df"] = None

if "vector_ready" not in st.session_state:
    st.session_state["vector_ready"] = False

if "quiz_questions" not in st.session_state:
    st.session_state["quiz_questions"] = []

if "quiz_text" not in st.session_state:
    st.session_state["quiz_text"] = ""

if "quiz_sources" not in st.session_state:
    st.session_state["quiz_sources"] = []

if "quiz_submitted" not in st.session_state:
    st.session_state["quiz_submitted"] = False

if "quiz_answers" not in st.session_state:
    st.session_state["quiz_answers"] = {}

if "quiz_warnings" not in st.session_state:
    st.session_state["quiz_warnings"] = []

if "quiz_id" not in st.session_state:
    st.session_state["quiz_id"] = 0

if "used_quiz_words" not in st.session_state:
    st.session_state["used_quiz_words"] = []


def add_vocab_to_session(new_vocab_df):
    if new_vocab_df is None or new_vocab_df.empty:
        return 0

    new_vocab_df = new_vocab_df.copy()
    new_vocab_df["word_key"] = new_vocab_df["word"].astype(str).str.lower().str.strip()

    current_df = st.session_state["vocab_df"]
    if current_df is None or current_df.empty:
        st.session_state["vocab_df"] = new_vocab_df.drop(columns=["word_key"])
        st.session_state["vector_ready"] = False
        return len(new_vocab_df)

    current_df = current_df.copy()
    current_keys = set(current_df["word"].astype(str).str.lower().str.strip())
    rows_to_add = new_vocab_df[~new_vocab_df["word_key"].isin(current_keys)].drop(
        columns=["word_key"]
    )

    if rows_to_add.empty:
        return 0

    st.session_state["vocab_df"] = pd.concat(
        [current_df, rows_to_add],
        ignore_index=True,
    )
    st.session_state["vector_ready"] = False
    return len(rows_to_add)


def ensure_vector_ready(force=False):
    if st.session_state["vector_ready"] and not force:
        return True

    vocab_df = st.session_state["vocab_df"]
    if vocab_df is None or vocab_df.empty:
        st.warning("請先加入單字。")
        return False

    try:
        with st.spinner("正在更新單字庫..."):
            build_vector_db(vocab_df)
        st.session_state["vector_ready"] = True
        st.success("單字庫建立完成")
        return True
    except Exception as e:
        st.error(f"建立單字庫失敗：{e}")
        return False


def should_use_direct_vocab_answer(question):
    question_text = question.strip()
    question_lower = question_text.lower()

    direct_keywords = [
        "翻譯",
        "意思",
        "中文",
        "怎麼說",
        "是什麼",
        "meaning",
        "mean",
        "translate",
        "definition",
        "define",
    ]

    if any(keyword in question_lower for keyword in direct_keywords):
        return True

    normalized = re.sub(r"[?？。！!，,：:\s]+", " ", question_text).strip()
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z '\-]*", normalized))


def find_direct_vocab_answer(question, vocab_df):
    if vocab_df is None or vocab_df.empty:
        return ""

    if not should_use_direct_vocab_answer(question):
        return ""

    question_lower = question.lower()
    matches = []

    for _, row in vocab_df.iterrows():
        word = str(row.get("word", "")).strip()
        if not word:
            continue

        word_lower = word.lower()
        pattern = r"(?<![A-Za-z])" + re.escape(word_lower) + r"(?![A-Za-z])"
        if re.search(pattern, question_lower):
            matches.append((len(word_lower), row))

    if not matches:
        return ""

    _, row = sorted(matches, key=lambda item: item[0], reverse=True)[0]
    word = str(row.get("word", "")).strip()
    pos = str(row.get("part_of_speech", "")).strip()
    zh_meaning = to_traditional_chinese(str(row.get("zh_meaning", "")).strip())
    en_definition = str(row.get("en_definition", "")).strip()
    source_files = str(row.get("source_files", row.get("source_file", ""))).strip()

    lines = [f"**{word}**"]

    if pos:
        lines.append(f"詞性：{pos}")
    if zh_meaning:
        lines.append(f"中文意思：{zh_meaning}")
    if en_definition:
        lines.append(f"英文解釋：{en_definition}")
    if source_files:
        lines.append(f"來源：{source_files}")

    return "\n\n".join(lines)


with st.sidebar:
    st.header("設定")
    show_advanced = st.checkbox("顯示進階資訊", value=False)

    if show_advanced:
        top_k = st.slider(
            "檢索筆數",
            min_value=20,
            max_value=100,
            value=60,
        )
    else:
        top_k = 60

    if st.button("重置練習紀錄"):
        st.session_state["used_quiz_words"] = []
        st.success("已重置練習紀錄。")

    if show_advanced:
        api_key_status = "已設定" if os.getenv("OPENAI_API_KEY") else "未設定"
        st.write(f"OPENAI API 金鑰：{api_key_status}")
        st.caption(f"已避免重複用字：{len(st.session_state['used_quiz_words'])}")


tab_upload, tab_ask, tab_quiz = st.tabs(
    ["1. 建立單字庫", "2. 單字問答", "3. 多益練習題"]
)


with tab_upload:
    st.subheader("建立單字庫")

    st.write("直接輸入單字，或上傳 txt 檔。")

    manual_vocab_text = st.text_area(
        "直接輸入單字",
        placeholder="""invoice=n. 發票
receipt=n. 收據
applicant=n. 申請者
reschedule=v. 重新安排時間
abide by=v. 遵守""",
        height=150,
    )

    st.caption("格式：一行一個單字，例如 `invoice=n. 發票`。")

    if st.button("讀取輸入內容", type="primary", use_container_width=True):
        if not manual_vocab_text.strip():
            st.warning("請先輸入單字。")
        else:
            merged = merge_vocab_frames(
                [parse_vocab_text(manual_vocab_text, "手動輸入")]
            )
            added_count = add_vocab_to_session(merged)
            if added_count:
                st.success(f"已加入 {added_count} 筆單字")
                st.info("送出問題或產生練習題時，系統會自動更新單字庫。")
            else:
                st.info("沒有新單字可加入")

    uploaded_files = st.file_uploader(
        "上傳單字 txt 檔",
        type=["txt"],
        accept_multiple_files=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("讀取上傳檔案", use_container_width=True):
            if not uploaded_files:
                st.warning("請先上傳 txt 檔。")
            else:
                frames = []
                for uploaded_file in uploaded_files:
                    text = read_uploaded_text(uploaded_file)
                    frames.append(parse_vocab_text(text, uploaded_file.name))

                merged = merge_vocab_frames(frames)
                added_count = add_vocab_to_session(merged)
                if added_count:
                    st.success(f"已加入 {added_count} 筆單字")
                    st.info("送出問題或產生練習題時，系統會自動更新單字庫。")
                else:
                    st.info("沒有新單字可加入")

    with col2:
        if st.button("讀取 local_data 預設檔案", use_container_width=True):
            try:
                merged, missing = load_local_vocab_files("local_data")
                added_count = add_vocab_to_session(merged)
                if added_count:
                    st.success(f"已加入 {added_count} 筆單字")
                    st.info("送出問題或產生練習題時，系統會自動更新單字庫。")
                else:
                    st.info("沒有新單字可加入")

                if missing and show_advanced:
                    st.warning(f"以下檔案不存在，已略過：{', '.join(missing)}")

            except Exception as e:
                st.error(str(e))

    vocab_df = st.session_state["vocab_df"]

    if vocab_df is not None:
        st.metric("單字數", len(vocab_df))

        if st.session_state["vector_ready"]:
            st.success("單字庫可使用")
        else:
            st.info("單字已讀取，尚未更新到問答系統。送出問題或產生練習題時會自動更新。")

        if show_advanced:
            with st.expander("查看單字資料"):
                st.dataframe(vocab_df.head(30), use_container_width=True)
                c1, c2, c3 = st.columns(3)
                c1.metric("合併後單字數", len(vocab_df))
                c2.metric("有中文解釋", int((vocab_df["zh_meaning"] != "").sum()))
                c3.metric("有英文解釋", int((vocab_df["en_definition"] != "").sum()))

        st.markdown("---")

        if st.button("建立/更新單字庫", type="primary"):
            ensure_vector_ready(force=True)
    else:
        st.info("請先讀取單字資料。")


with tab_ask:
    st.subheader("單字問答")

    if st.session_state["vocab_df"] is None:
        st.warning("請先加入單字。")
    elif not st.session_state["vector_ready"]:
        st.info("單字庫尚未更新，送出問題時會自動更新。")

    question = st.text_input(
        "想查什麼單字？",
        placeholder="例如：abide by 是什麼意思？請整理 accounting 相關單字。",
    )

    if st.button("送出問題", type="primary"):
        if not question.strip():
            st.warning("請先輸入問題。")
        elif not ensure_vector_ready():
            pass
        else:
            try:
                direct_answer = find_direct_vocab_answer(
                    question,
                    st.session_state["vocab_df"],
                )

                if direct_answer:
                    results = []
                    answer = direct_answer
                else:
                    with st.spinner("正在查詢..."):
                        results = retrieve_context(question, top_k=top_k)
                        answer = generate_answer(question, results)

                st.markdown("### 回答")
                st.markdown(to_traditional_chinese(answer))

                if results:
                    with st.expander("查看回答依據"):
                        for i, item in enumerate(results, start=1):
                            word = item["metadata"].get("word", "")
                            if show_advanced:
                                with st.expander(f"{i}. {word}"):
                                    st.write(f"檢索距離：{item['distance']}")
                                    st.json(item["metadata"])
                                    st.code(item["document"])
                            else:
                                st.write(f"{i}. {word}")

            except Exception as e:
                st.error(f"問答失敗：{e}")


with tab_quiz:
    st.subheader("多益練習題")
    st.info("題目由系統生成，非 TOEIC 官方題目")

    if st.session_state["vocab_df"] is None:
        st.warning("請先加入單字。")
    elif not st.session_state["vector_ready"]:
        st.info("單字庫尚未更新，產生練習題時會自動更新。")

    topic = st.text_input(
        "練習主題",
        placeholder="例如：business office finance meeting travel",
    )

    num_questions = st.radio("題數", [3, 5], index=1, horizontal=True)

    if st.button("產生練習題", type="primary"):
        if not ensure_vector_ready():
            pass
        else:
            try:
                with st.spinner("正在產生題目..."):
                    quiz_questions, results, warnings, used_words_this_round = generate_quiz(
                        topic=topic,
                        num_questions=num_questions,
                        top_k=max(top_k, num_questions * 12),
                        used_words=st.session_state["used_quiz_words"],
                    )

                st.session_state["used_quiz_words"].extend(used_words_this_round)
                st.session_state["used_quiz_words"] = list(
                    dict.fromkeys(st.session_state["used_quiz_words"])
                )
                st.session_state["quiz_text"] = ""
                st.session_state["quiz_questions"] = quiz_questions
                st.session_state["quiz_sources"] = results
                st.session_state["quiz_warnings"] = warnings
                st.session_state["quiz_submitted"] = False
                st.session_state["quiz_answers"] = {}
                st.session_state["quiz_id"] += 1
                st.success("題目已生成")

            except Exception as e:
                st.error(f"測驗生成失敗：{e}")

    quiz_questions = st.session_state["quiz_questions"]

    if quiz_questions:
        st.markdown("### 測驗題")
        st.caption(f"共 {len(quiz_questions)} 題，請全部作答後再送出。")

        if show_advanced and st.session_state["quiz_warnings"]:
            st.warning("進階檢查提醒：")
            for warning in st.session_state["quiz_warnings"]:
                st.write(f"- {warning}")

        with st.form("quiz_form"):
            selected_answers = {}

            for i, question in enumerate(quiz_questions, start=1):
                st.markdown(f"**{i}. {question['question']}**")
                option_labels = [
                    f"({letter}) {text}"
                    for letter, text in question["options"].items()
                ]
                selected_label = st.radio(
                    "選擇最適合的答案",
                    option_labels,
                    index=None,
                    key=f"quiz_{st.session_state['quiz_id']}_answer_{i}",
                    label_visibility="collapsed",
                )
                selected_answers[i] = selected_label[1] if selected_label else ""

            submitted = st.form_submit_button("送出答案", type="primary")

        if submitted:
            missing_numbers = [
                str(i) for i, answer in selected_answers.items() if not answer
            ]

            if missing_numbers:
                st.warning(f"請先完成第 {', '.join(missing_numbers)} 題。")
            else:
                st.session_state["quiz_answers"] = selected_answers
                st.session_state["quiz_submitted"] = True

        if st.session_state["quiz_submitted"]:
            answers = st.session_state["quiz_answers"]
            score = sum(
                1
                for i, question in enumerate(quiz_questions, start=1)
                if answers.get(i) == question["answer"]
            )

            st.markdown("### 測驗結果")
            st.metric("得分", f"{score} / {len(quiz_questions)}")

            for i, question in enumerate(quiz_questions, start=1):
                user_answer = answers.get(i, "")
                correct_answer = question["answer"]
                is_correct = user_answer == correct_answer
                status = "答對" if is_correct else "答錯"

                with st.expander(f"第 {i} 題｜{status}", expanded=not is_correct):
                    st.write(f"你的答案：{user_answer or '未作答'}")
                    st.write(f"正確答案：({correct_answer}) {question['options'][correct_answer]}")
                    if question.get("word"):
                        st.write(f"目標單字：{question['word']}")
                    if question.get("explanation"):
                        st.write(f"解析：{question['explanation']}")
                    if question.get("source"):
                        st.write(f"來源單字：{question['source']}")

        with st.expander("查看出題依據"):
            for i, item in enumerate(st.session_state["quiz_sources"], start=1):
                word = item["metadata"].get("word", "")
                if show_advanced:
                    with st.expander(f"{i}. {word}"):
                        st.write(f"檢索距離：{item['distance']}")
                        st.json(item["metadata"])
                        st.code(item["document"])
                else:
                    st.write(f"{i}. {word}")
