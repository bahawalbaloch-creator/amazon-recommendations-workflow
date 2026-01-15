from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from code_interpreter_system_prompt import SYSTEM_PROMPT, ASSISTANT_PROMPT

# Load environment variables early.
load_dotenv()

MODEL_DEFAULT = os.getenv("OPENAI_ASSISTANT_MODEL", "gpt-4o-mini")

# Tool definition for get_campaign_summary
GET_CAMPAIGN_SUMMARY_TOOL = {
    "type": "function",
    "function": {
        "name": "get_campaign_summary",
        "description": "Retrieve campaign data from the database including campaign performance, campaign info, keyword performance with bids, and computed metrics (CTR, CPC, ROAS, score). Use this when the user asks for recommendations, optimization, or analysis of a specific campaign.",
        "parameters": {
            "type": "object",
            "properties": {
                "campaign_name": {
                    "type": "string",
                    "description": "The exact name of the campaign to retrieve data for. This should be extracted from the user's query."
                }
            },
            "required": ["campaign_name"]
        }
    }
}


@st.cache_resource
def get_openai_client() -> OpenAI:
    """
    Create a cached OpenAI client.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key:
        st.error("OPENAI_API_KEY is not set. Please add it to your .env file.")
        st.stop()
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    return OpenAI(**client_kwargs) 


def ensure_state() -> None:
    """Initialize Streamlit session state keys."""
    defaults = {
        "messages": [],
        "thread_id": None,
        "assistant_id": None,
        "assistant_prompt": ASSISTANT_PROMPT,
        "db_path": os.getenv("DB_PATH", "ads_data.db"),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def create_or_update_assistant(
    client: OpenAI, instructions: str, model: str
) -> str:
    """
    Create a new assistant when instructions change, otherwise reuse.
    If the prompt changes, the conversation thread is reset to avoid stale context.
    """
    prompt_changed = instructions.strip() != (st.session_state.get("assistant_prompt") or "").strip()
    if st.session_state.get("assistant_id") is None or prompt_changed:
        assistant = client.beta.assistants.create(
            name="Amazon Campaign Assistant",
            instructions=instructions.strip(),
            model=model,
            tools=[GET_CAMPAIGN_SUMMARY_TOOL],
        )
        st.session_state.assistant_id = assistant.id
        st.session_state.assistant_prompt = instructions.strip()
        # reset thread when prompt changes so context matches the new brief
        st.session_state.thread_id = None
    return st.session_state.assistant_id


def ensure_thread(client: OpenAI) -> str:
    if st.session_state.get("thread_id") is None:
        thread = client.beta.threads.create()
        st.session_state.thread_id = thread.id
    return st.session_state.thread_id


def get_campaign_summary(db_path: str, campaign_name: str) -> Optional[Dict[str, Any]]:
    """
    Returns a campaign summary:
    - campaign performance
    - campaign info
    - aggregated keyword performance
    - bid from keyword_ad_performance
    - score out of 10
    """
    summary = {}

    try:
        conn = sqlite3.connect(db_path)

        # --- Campaign performance ---
        campaign_perf = pd.read_sql_query(
            "SELECT * FROM CampaignPerformance WHERE campaign_name = ?",
            conn,
            params=(campaign_name,)
        )
        summary['campaign_performance'] = campaign_perf.to_dict(orient='records')[0] if not campaign_perf.empty else None

        # --- Campaign info ---
        campaign_info = pd.read_sql_query(
            "SELECT * FROM Campaign WHERE campaign_name = ?",
            conn,
            params=(campaign_name,)
        )
        summary['campaign_info'] = campaign_info.to_dict(orient='records')[0] if not campaign_info.empty else None

        # --- Keyword performance ---
        kw_perf = pd.read_sql_query(
            "SELECT * FROM KeywordPerformance WHERE campaign_name = ?",
            conn,
            params=(campaign_name,)
        )
        if kw_perf.empty:
            summary['keyword_performance'] = []
            summary['keywords'] = []
            conn.close()
            return summary

        # --- Aggregate metrics by ad_group_name + customer_search_term ---
        kw_agg = kw_perf.groupby(['ad_group_name', 'customer_search_term']).agg(
            impressions=pd.NamedAgg(column='impressions', aggfunc='sum'),
            clicks=pd.NamedAgg(column='clicks', aggfunc='sum'),
            spend=pd.NamedAgg(column='spend', aggfunc='sum'),
            sales=pd.NamedAgg(column='7_day_total_sales', aggfunc='sum'),
            orders=pd.NamedAgg(column='7_day_total_orders_#', aggfunc='sum')
        ).reset_index()

        # --- Compute CTR, CPC, ROAS ---
        kw_agg['ctr'] = kw_agg['clicks'] / kw_agg['impressions'].replace(0, 1)
        kw_agg['cpc'] = kw_agg['spend'] / kw_agg['clicks'].replace(0, 1)
        kw_agg['roas'] = kw_agg['sales'] / kw_agg['spend'].replace(0, 1)

        # --- Merge bid from keyword_ad_performance ---
        ad_perf = pd.read_sql_query(
            """
            SELECT campaign_name_ AS campaign_name,
                   ad_group_name_ AS ad_group_name,
                   keyword_text AS customer_search_term,
                   bid,
                   ad_group_default_bid_ AS default_bid
            FROM keyword_ad_performance
            WHERE campaign_name_ = ?
            """,
            conn,
            params=(campaign_name,)
        )

        # Fill missing bid with default bid
        ad_perf['bid'] = ad_perf['bid'].fillna(ad_perf['default_bid'])

        # Merge bids into aggregated keyword metrics
        kw_agg = kw_agg.merge(
            ad_perf[['ad_group_name', 'customer_search_term', 'bid']],
            on=['ad_group_name', 'customer_search_term'],
            how='left'
        )

        # --- Compute score out of 10 ---
        def compute_score(row):
            score = 0
            # CTR (0-4 points)
            if row['ctr'] > 0.1:      score += 4
            elif row['ctr'] > 0.05:   score += 3
            elif row['ctr'] > 0.02:   score += 2
            elif row['ctr'] > 0:      score += 1

            # ROAS (0-3 points)
            if row['roas'] > 5:       score += 3
            elif row['roas'] > 3:     score += 2
            elif row['roas'] > 1:     score += 1

            # Orders (0-3 points)
            if row['orders'] > 20:    score += 3
            elif row['orders'] > 10:  score += 2
            elif row['orders'] > 0:   score += 1

            return min(score, 10)

        kw_agg['score'] = kw_agg.apply(compute_score, axis=1)

        # --- Convert to JSON-friendly format ---
        summary['keyword_performance'] = kw_agg.to_dict(orient='records')
        summary['keywords'] = kw_agg['customer_search_term'].tolist()

        conn.close()
        return summary

    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
        return None
    except Exception as e:  # noqa: BLE001
        st.error(f"Error retrieving campaign summary: {e}")
        return None


def handle_tool_calls(
    client: OpenAI,
    thread_id: str,
    run_id: str,
    tool_calls: List[Any],
    db_path: str,
) -> None:
    """
    Execute tool calls and submit results back to the assistant.
    """
    tool_outputs = []
    
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        
        if function_name == "get_campaign_summary":
            campaign_name = arguments.get("campaign_name", "")
            st.info(f"🔍 Retrieving data for campaign: {campaign_name}")
            
            result = get_campaign_summary(db_path, campaign_name)
            
            if result:
                st.success(f"✅ Retrieved data for campaign: {campaign_name}")
                output = json.dumps(result, indent=2, default=str)
            else:
                output = json.dumps({
                    "error": f"Could not retrieve data for campaign: {campaign_name}",
                    "suggestion": "Please check the campaign name spelling or try a different campaign."
                })
            
            tool_outputs.append({
                "tool_call_id": tool_call.id,
                "output": output
            })
        else:
            tool_outputs.append({
                "tool_call_id": tool_call.id,
                "output": json.dumps({"error": f"Unknown function: {function_name}"})
            })
    
    # Submit tool outputs
    client.beta.threads.runs.submit_tool_outputs(
        thread_id=thread_id,
        run_id=run_id,
        tool_outputs=tool_outputs
    )


def run_assistant_with_tools(
    client: OpenAI,
    thread_id: str,
    assistant_id: str,
    instructions: str,
    placeholder,
    db_path: str,
) -> Optional[str]:
    """
    Run the assistant and handle tool calls. Returns the final response.
    """
    accumulated = ""
    
    try:
        # Create the run
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id,
            instructions=instructions,
        )
        
        # Poll for completion
        while True:
            run = client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id
            )
            
            if run.status == "completed":
                break
            elif run.status == "requires_action":
                # Handle tool calls
                tool_calls = run.required_action.submit_tool_outputs.tool_calls
                handle_tool_calls(client, thread_id, run.id, tool_calls, db_path)
                placeholder.markdown("🔄 Processing campaign data... ▌")
            elif run.status in ["failed", "cancelled", "expired"]:
                placeholder.error(f"Run failed with status: {run.status}")
                return None
            else:
                # Still running, show loading state
                placeholder.markdown("⏳ Analyzing... ▌")
                time.sleep(0.5)
        
        # Get the final messages
        messages = client.beta.threads.messages.list(
            thread_id=thread_id,
            order="desc",
            limit=1
        )
        
        if messages.data:
            for content in messages.data[0].content:
                if content.type == "text":
                    accumulated = content.text.value
        
        placeholder.markdown(accumulated)
        return accumulated
        
    except Exception as exc:  # noqa: BLE001
        placeholder.error(f"Run failed: {exc}")
        return None


def main() -> None:
    st.set_page_config(page_title="Amazon Campaign Assistant", layout="wide")
    ensure_state()
    if st.session_state.assistant_prompt is None:
        st.session_state.assistant_prompt = SYSTEM_PROMPT
    client = get_openai_client()

    st.title("Amazon Campaign Assistant")
    st.write(
        "Chat with an AI assistant that can retrieve and analyze campaign data from the database."
    )
    thread_label = st.session_state.thread_id or "Not created yet"
    st.info(f"Thread ID: `{thread_label}`", icon="🧵")

    with st.sidebar:
        st.subheader("Settings")
        st.caption("System prompt is configured in code.")
        st.caption(f"Model: {MODEL_DEFAULT}")
        
        db_path = st.text_input(
            "Database Path",
            value=st.session_state.db_path,
            help="Path to the SQLite database file (ads_data.db)",
        )
        st.session_state.db_path = db_path

        if st.button("Reset conversation"):
            st.session_state.messages = []
            st.session_state.thread_id = None
            st.toast("Conversation reset.")

    # Render previous turns
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask about a campaign or request an analysis")
    if user_input:
        assistant_id = create_or_update_assistant(client, SYSTEM_PROMPT, MODEL_DEFAULT)
        thread_id = ensure_thread(client)

        # Record user message locally
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Send the message to the thread
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=user_input,
        )

        # Run assistant with tool calling support
        with st.chat_message("assistant"):
            placeholder = st.empty()
            response = run_assistant_with_tools(
                client=client,
                thread_id=thread_id,
                assistant_id=assistant_id,
                instructions=SYSTEM_PROMPT,
                placeholder=placeholder,
                db_path=st.session_state.db_path,
            )

        if response:
            st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
