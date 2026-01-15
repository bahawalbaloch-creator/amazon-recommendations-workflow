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

# Tool definition for get_keyword_recommendations
GET_KEYWORD_RECOMMENDATIONS_TOOL = {
    "type": "function",
    "function": {
        "name": "get_keyword_recommendations",
        "description": "Analyze keyword performance to find new keywords to ADD (from high-performing customer search terms) and keywords to REMOVE (targeting keywords causing losses). Use this when: 1) User explicitly asks for keyword recommendations, 2) All keywords in campaign analysis are performing poorly, 3) User wants to discover new keyword opportunities.",
        "parameters": {
            "type": "object",
            "properties": {
                "campaign_name": {
                    "type": "string",
                    "description": "The exact name of the campaign to analyze keywords for."
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
            tools=[GET_CAMPAIGN_SUMMARY_TOOL, GET_KEYWORD_RECOMMENDATIONS_TOOL],
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
    - campaign performance (includes average_time_in_budget - % of time campaign had budget)
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


def get_keyword_recommendations(db_path: str, campaign_name: str) -> Optional[Dict[str, Any]]:
    """
    Analyze KeywordPerformance to find:
    - 5 new keywords to ADD (high-performing customer_search_terms not in targeting)
    - 5 keywords to REMOVE (targeting keywords causing losses)
    
    Returns detailed performance metrics and reasoning for each recommendation.
    """
    recommendations = {
        "campaign_name": campaign_name,
        "keywords_to_add": [],
        "keywords_to_remove": [],
        "summary": {}
    }

    try:
        conn = sqlite3.connect(db_path)

        # Get all keyword performance data for the campaign
        kw_perf = pd.read_sql_query(
            """
            SELECT 
                targeting,
                customer_search_term,
                impressions,
                clicks,
                spend,
                7_day_total_sales as sales,
                "7_day_total_orders_#" as orders,
                cost_per_click as cpc,
                click_thru_rate as ctr
            FROM KeywordPerformance 
            WHERE campaign_name = ?
            """,
            conn,
            params=(campaign_name,)
        )

        if kw_perf.empty:
            conn.close()
            return {
                "error": f"No keyword data found for campaign: {campaign_name}",
                "suggestion": "Check campaign name or ensure data exists in KeywordPerformance table."
            }

        # Convert percentages if needed (CTR might be stored as decimal or percentage)
        if kw_perf['ctr'].max() < 1:
            kw_perf['ctr'] = kw_perf['ctr'] * 100

        # Calculate ACOS and ROAS
        kw_perf['acos'] = (kw_perf['spend'] / kw_perf['sales'].replace(0, 0.001)) * 100
        kw_perf['roas'] = kw_perf['sales'] / kw_perf['spend'].replace(0, 0.001)

        # Handle infinite values
        kw_perf['acos'] = kw_perf['acos'].replace([float('inf'), -float('inf')], 999)
        kw_perf['roas'] = kw_perf['roas'].replace([float('inf'), -float('inf')], 0)

        # ============================================
        # KEYWORDS TO ADD: High-performing customer search terms
        # ============================================
        # Aggregate by customer_search_term to find winning search terms
        search_term_agg = kw_perf.groupby('customer_search_term').agg(
            impressions=('impressions', 'sum'),
            clicks=('clicks', 'sum'),
            spend=('spend', 'sum'),
            sales=('sales', 'sum'),
            orders=('orders', 'sum')
        ).reset_index()

        # Recalculate metrics after aggregation
        search_term_agg['ctr'] = (search_term_agg['clicks'] / search_term_agg['impressions'].replace(0, 1)) * 100
        search_term_agg['cpc'] = search_term_agg['spend'] / search_term_agg['clicks'].replace(0, 1)
        search_term_agg['acos'] = (search_term_agg['spend'] / search_term_agg['sales'].replace(0, 0.001)) * 100
        search_term_agg['roas'] = search_term_agg['sales'] / search_term_agg['spend'].replace(0, 0.001)
        search_term_agg['cvr'] = (search_term_agg['orders'] / search_term_agg['clicks'].replace(0, 1)) * 100

        # Handle infinite values
        search_term_agg['acos'] = search_term_agg['acos'].replace([float('inf'), -float('inf')], 999)
        search_term_agg['roas'] = search_term_agg['roas'].replace([float('inf'), -float('inf')], 0)

        # Get existing targeting keywords
        existing_targets = set(kw_perf['targeting'].dropna().str.lower().unique())

        # Filter: search terms that are NOT already targeted and have good performance
        # Criteria: impressions >= 10, has orders, ROAS > 1.5
        potential_adds = search_term_agg[
            (search_term_agg['impressions'] >= 10) &
            (search_term_agg['orders'] > 0) &
            (search_term_agg['roas'] > 1.5) &
            (~search_term_agg['customer_search_term'].str.lower().isin(existing_targets))
        ].copy()

        # Score for ranking: prioritize ROAS, orders, and impressions
        potential_adds['add_score'] = (
            potential_adds['roas'] * 2 +
            potential_adds['orders'] * 3 +
            (potential_adds['impressions'] / 100)
        )

        # Get top 5 keywords to add
        top_adds = potential_adds.nlargest(5, 'add_score')

        for _, row in top_adds.iterrows():
            reasoning = []
            if row['roas'] > 3:
                reasoning.append(f"Excellent ROAS of {row['roas']:.2f}x")
            elif row['roas'] > 2:
                reasoning.append(f"Good ROAS of {row['roas']:.2f}x")
            else:
                reasoning.append(f"Positive ROAS of {row['roas']:.2f}x")
            
            if row['orders'] >= 5:
                reasoning.append(f"Strong conversion with {int(row['orders'])} orders")
            else:
                reasoning.append(f"Converting with {int(row['orders'])} orders")
            
            if row['acos'] < 25:
                reasoning.append(f"Low ACOS at {row['acos']:.1f}%")
            
            recommendations["keywords_to_add"].append({
                "keyword": row['customer_search_term'],
                "impressions": int(row['impressions']),
                "clicks": int(row['clicks']),
                "spend": round(row['spend'], 2),
                "sales": round(row['sales'], 2),
                "orders": int(row['orders']),
                "ctr": round(row['ctr'], 2),
                "cpc": round(row['cpc'], 2),
                "acos": round(row['acos'], 2),
                "roas": round(row['roas'], 2),
                "cvr": round(row['cvr'], 2),
                "reasoning": " | ".join(reasoning),
                "recommendation": "ADD as Exact Match targeting keyword"
            })

        # ============================================
        # KEYWORDS TO REMOVE: Targeting keywords causing losses
        # ============================================
        # Aggregate by targeting keyword
        targeting_agg = kw_perf.groupby('targeting').agg(
            impressions=('impressions', 'sum'),
            clicks=('clicks', 'sum'),
            spend=('spend', 'sum'),
            sales=('sales', 'sum'),
            orders=('orders', 'sum')
        ).reset_index()

        # Recalculate metrics
        targeting_agg['ctr'] = (targeting_agg['clicks'] / targeting_agg['impressions'].replace(0, 1)) * 100
        targeting_agg['cpc'] = targeting_agg['spend'] / targeting_agg['clicks'].replace(0, 1)
        targeting_agg['acos'] = (targeting_agg['spend'] / targeting_agg['sales'].replace(0, 0.001)) * 100
        targeting_agg['roas'] = targeting_agg['sales'] / targeting_agg['spend'].replace(0, 0.001)
        targeting_agg['cvr'] = (targeting_agg['orders'] / targeting_agg['clicks'].replace(0, 1)) * 100

        # Handle infinite values
        targeting_agg['acos'] = targeting_agg['acos'].replace([float('inf'), -float('inf')], 999)
        targeting_agg['roas'] = targeting_agg['roas'].replace([float('inf'), -float('inf')], 0)

        # Filter: targeting keywords that are bleeding money
        # Criteria: impressions >= 10 AND (spend > $5 with no sales OR ACOS > 50%)
        potential_removes = targeting_agg[
            (targeting_agg['impressions'] >= 10) &
            (
                ((targeting_agg['spend'] > 5) & (targeting_agg['sales'] == 0)) |
                (targeting_agg['acos'] > 50)
            )
        ].copy()

        # Score for ranking: prioritize high spend with low/no return
        potential_removes['remove_score'] = (
            potential_removes['spend'] * 2 +
            (potential_removes['acos'] / 10) -
            (potential_removes['orders'] * 10)
        )

        # Get top 5 keywords to remove
        top_removes = potential_removes.nlargest(5, 'remove_score')

        for _, row in top_removes.iterrows():
            reasoning = []
            if row['sales'] == 0 and row['spend'] > 0:
                reasoning.append(f"💸 Spent ${row['spend']:.2f} with ZERO sales")
                action = "PAUSE immediately - bleeding money"
            elif row['acos'] > 100:
                reasoning.append(f"🔴 ACOS of {row['acos']:.1f}% (spending more than earning)")
                action = "PAUSE or significantly reduce bid"
            elif row['acos'] > 50:
                reasoning.append(f"⚠️ High ACOS of {row['acos']:.1f}%")
                action = "Reduce bid by 30-50% or add as negative"
            else:
                reasoning.append(f"Underperforming with ACOS {row['acos']:.1f}%")
                action = "Review and consider pausing"
            
            if row['clicks'] > 10 and row['orders'] == 0:
                reasoning.append(f"No conversions from {int(row['clicks'])} clicks")
            
            if row['ctr'] < 0.5 and row['impressions'] > 100:
                reasoning.append(f"Very low CTR ({row['ctr']:.2f}%) - poor relevance")

            recommendations["keywords_to_remove"].append({
                "keyword": row['targeting'],
                "impressions": int(row['impressions']),
                "clicks": int(row['clicks']),
                "spend": round(row['spend'], 2),
                "sales": round(row['sales'], 2),
                "orders": int(row['orders']),
                "ctr": round(row['ctr'], 2),
                "cpc": round(row['cpc'], 2),
                "acos": round(row['acos'], 2) if row['acos'] < 999 else "∞ (no sales)",
                "roas": round(row['roas'], 2),
                "cvr": round(row['cvr'], 2),
                "reasoning": " | ".join(reasoning),
                "recommendation": action
            })

        # Summary statistics
        recommendations["summary"] = {
            "total_search_terms_analyzed": len(search_term_agg),
            "total_targeting_keywords_analyzed": len(targeting_agg),
            "keywords_recommended_to_add": len(recommendations["keywords_to_add"]),
            "keywords_recommended_to_remove": len(recommendations["keywords_to_remove"]),
            "potential_monthly_savings": round(top_removes['spend'].sum(), 2) if not top_removes.empty else 0,
            "note": "Keywords to ADD are customer search terms converting well but not explicitly targeted. Keywords to REMOVE are targeting keywords that drain budget without returns."
        }

        conn.close()
        return recommendations

    except sqlite3.Error as e:
        return {"error": f"Database error: {e}"}
    except Exception as e:  # noqa: BLE001
        return {"error": f"Error analyzing keywords: {e}"}


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
        
        elif function_name == "get_keyword_recommendations":
            campaign_name = arguments.get("campaign_name", "")
            st.info(f"🔎 Analyzing keywords for campaign: {campaign_name}")
            
            result = get_keyword_recommendations(db_path, campaign_name)
            
            if result and "error" not in result:
                adds = len(result.get("keywords_to_add", []))
                removes = len(result.get("keywords_to_remove", []))
                st.success(f"✅ Found {adds} keywords to ADD, {removes} keywords to REMOVE")
                output = json.dumps(result, indent=2, default=str)
            else:
                output = json.dumps(result if result else {
                    "error": f"Could not analyze keywords for campaign: {campaign_name}",
                    "suggestion": "Please check the campaign name or ensure data exists."
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
