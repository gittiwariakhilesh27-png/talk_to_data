import streamlit as st
import pandas as pd
from decimal import Decimal
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Any
from dotenv import load_dotenv
import json

# ---------------- CONFIG ---------------- #
load_dotenv(override=True)
TABLE_NAME = "data_table"
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------------- STATE ---------------- #
class Text2SQLState(TypedDict, total=False):
    ask: str
    query: str
    conversation: list[str] 
    clarified_ask: str
    needs_clarification: bool
    clarification_question: str
    output: list[dict[str, Any]]
    insight: str

# ---------------- UTILITIES ---------------- #
def load_file(uploaded_file) -> pd.DataFrame:
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file type")

def create_sqlite_table(df: pd.DataFrame):
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    df.to_sql(TABLE_NAME, engine, index=False, if_exists="replace")
    return engine

def dataframe_schema(df: pd.DataFrame) -> dict:
    return {col: str(df[col].dtype) for col in df.columns}

# ---------------- AGENTS ---------------- #
def clarification_agent(state: Text2SQLState, df: pd.DataFrame) -> Text2SQLState:
    schema = dataframe_schema(df)
    conversation_text = "\n".join(state.get("conversation", []))

    prompt = f"""
You are a clarification agent for Text-to-SQL.

Schema:
{schema}

Conversation so far:
{conversation_text}

Rules:
- Decide if enough information exists to write SQL
- If NOT, ask ONE precise clarification question
- If YES, respond CLEAR
- Do not repeat previous clarification questions

Output JSON ONLY:
{{
  "status": "CLEAR" | "CLARIFY",
  "question": "<clarification question or empty>"
}}
"""
    resp = llm.invoke([HumanMessage(content=prompt)]).content
    parsed = json.loads(resp)

    if parsed["status"] == "CLARIFY":
        return {**state, "needs_clarification": True, "clarification_question": parsed["question"]}
    return {**state, "needs_clarification": False, "clarified_ask": conversation_text}

def query_generator_agent(state: Text2SQLState, df: pd.DataFrame) -> Text2SQLState:
    schema = dataframe_schema(df)
    # st.write(schema)
    question = state.get("clarified_ask", state.get("ask", ""))

    prompt = f"""
You are a deterministic SQL generation engine for SQLLITE.

Rules:
1. Output ONLY raw SQL QUERY , Return ONLY ONE query , DO not return multiple queries.
2. Use ONLY this table name EXACTLY as written: `{TABLE_NAME}`
3. Use ONLY columns in schema,DO NOT put any special charachter
4. Fully qualify column names
5. ONLY Apply LOWER()/UPPER()/PROPER() function STRING comparison as per you judgement, DO NOT APPLY EVERYWHERE
6. Alias columns when using functions
7. Output ONLY raw SQL (no text, no markdown)
8. If question is impossible, output exactly: INVALID_QUESTION OR if columns mismatch says : COLUMN_NOT_PRESENT

Schema:
{schema}

Question:
{question}
"""
    resp = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    return {**state, "query": resp}

def output_reader(state: Text2SQLState, engine) -> Text2SQLState:
    query = state["query"].strip()
    # st.write(query)
    if query in ("INVALID_QUESTION", "COLUMN_NOT_PRESENT"):
        return {**state, "output": []}

    with engine.connect() as conn:
        result = conn.execute(text(query))
        columns = result.keys()
        rows = result.fetchall()

    output = []
    for row in rows:
        row_dict = {col: float(val) if isinstance(val, Decimal) else val for col, val in zip(columns, row)}
        output.append(row_dict)

    return {**state, "output": output}

def insight_generator_agent(state: Text2SQLState) -> Text2SQLState:
    if not state.get("output"):
        return {**state, "insight": "No data available to answer the question."}

    payload = {"question": state.get("ask", ""), "data": state["output"]}
    prompt = f"""
You are a business data analyst.

Rules:
- Use ONLY the data provided
- Do NOT guess
- Keep answer to 2–3 sentences
- Be professional

Input:
{payload}
"""
    insight = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    return {**state, "insight": insight}

# ---------------- GRAPH ---------------- #
workflow = StateGraph(Text2SQLState)
workflow.add_node("clarification_agent", clarification_agent)
workflow.add_node("query_generator_agent", query_generator_agent)
workflow.add_node("output_reader", output_reader)
workflow.add_node("insight_generator_agent", insight_generator_agent)
workflow.set_entry_point("clarification_agent")
workflow.add_conditional_edges("clarification_agent", lambda s: END if s.get("needs_clarification") else "query_generator_agent")
workflow.add_edge("query_generator_agent", "output_reader")
workflow.add_conditional_edges("output_reader", lambda s: END if not s.get("output") else "insight_generator_agent")
workflow.add_edge("insight_generator_agent", END)
app = workflow.compile()

# ---------------- STREAMLIT APP ---------------- #
st.title("Talk to Your Uploaded Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel file", ["csv", "xlsx"]
)

if uploaded_file:
    df = load_file(uploaded_file)
    df.columns= (df.columns.str.strip().str.replace(" ", "_", regex=False))
    engine = create_sqlite_table(df)

    # ---------------- CHAT MEMORY ----------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "pending_clarification" not in st.session_state:
        st.session_state.pending_clarification = False

    if "last_state" not in st.session_state:
        st.session_state.last_state = None

    # ---------------- RENDER HISTORY ----------------
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # ---------------- INPUT ----------------
    user_input = st.chat_input("Ask a question about your data")

    if user_input:
        # ---- USER ----
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )
        st.chat_message("user").write(user_input)

        # ---- BUILD CONVERSATION FOR AGENTS ----
        conversation = [
            f"{m['role'].capitalize()}: {m['content']}"
            for m in st.session_state.messages
        ]

        state = {
            "ask": user_input,
            "conversation": conversation
        }

        # ---- CLARIFICATION ----
        state = clarification_agent(state, df)

        if state.get("needs_clarification"):
            question = f"""Clarification Agent : {state["clarification_question"]}"""

            st.session_state.pending_clarification = True
            st.session_state.last_state = state

            st.session_state.messages.append(
                {"role": "assistant", "content": question}
            )
            st.chat_message("assistant").write(question)
            st.stop()

        # ---- QUERY → RESULT → INSIGHT ----
        state = query_generator_agent(state, df)
        state = output_reader(state, engine)
        state = insight_generator_agent(state)

        assistant_text = state.get("insight", "No insight generated.")

        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_text}
        )

        assistant = st.chat_message("assistant")
        assistant.write(assistant_text)

        assistant.subheader("Generated SQL")
        assistant.code(state.get("query"))

        assistant.subheader("Query Result")
        st.dataframe(pd.DataFrame(state.get("output", [])))
