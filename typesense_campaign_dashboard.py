"""
Amazon Ads Campaign Analytics Dashboard
Powered by Typesense for fast lookups and GPT for AI recommendations.

Run:
  streamlit run typesense_campaign_dashboard.py
"""

import logging
import os
import sys

import pandas as pd
import streamlit as st
import typesense
from dotenv import load_dotenv
from campaign_recommendations import generate_recommendations_streaming
from token_tracker import get_tracker

# Load environment variables from .env file
load_dotenv()

# ==================== LOGGING SETUP ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
logger.info("=" * 60)
logger.info("Amazon Ads Analytics Dashboard starting...")
logger.info("=" * 60)


# ==================== CONFIGURATION ====================

st.set_page_config(
    page_title="Amazon Ads Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for professional look
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #232F3E 0%, #37475A 100%);
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 600;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1rem;
    }
    
    /* Section headers */
    .section-header {
        background: #f8f9fa;
        padding: 0.75rem 1rem;
        border-left: 4px solid #FF9900;
        margin: 1.5rem 0 1rem 0;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 6px;
        font-weight: 500;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: #f8f9fa;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Dropdown full width */
    div[data-baseweb="select"] span {
        overflow: visible !important;
        text-overflow: initial !important;
        white-space: normal !important;
    }
    div[data-baseweb="select"] {
        max-width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)


# ==================== HELPER FUNCTIONS ====================

@st.cache_resource
def get_client():
    host = os.getenv("TYPESENSE_HOST", "localhost")
    port = int(os.getenv("TYPESENSE_PORT", "8108"))
    protocol = os.getenv("TYPESENSE_PROTOCOL", "http")
    logger.info(f"Connecting to Typesense at {protocol}://{host}:{port}")
    return typesense.Client({
        "nodes": [{"host": host, "port": port, "protocol": protocol}],
        "api_key": os.getenv("TYPESENSE_API_KEY", "mykey"),
        "connection_timeout_seconds": 5,
    })


def list_collections(client):
    logger.info("Fetching collections from Typesense...")
    cols = client.collections.retrieve()
    logger.info(f"Found {len(cols)} collections")
    return [c["name"] for c in cols]


def get_collection_stats(client):
    """Get stats for all collections."""
    logger.info("Fetching collection stats...")
    cols = client.collections.retrieve()
    stats = []
    for c in cols:
        stats.append({
            "name": c["name"],
            "num_documents": c.get("num_documents", 0),
            "num_fields": len(c.get("fields", [])),
            "created_at": c.get("created_at", "N/A"),
        })
    return stats


def get_collection_schema(client, collection_name: str):
    """Get detailed schema for a collection."""
    logger.info(f"Fetching schema for '{collection_name}'...")
    schema = client.collections[collection_name].retrieve()
    return schema


def run_typesense_query(client, collection_name: str, query: str, query_by: str, 
                         filter_by: str = "", per_page: int = 20):
    """Run a search query against a collection."""
    # Clean inputs
    query = query.strip() if query else "*"
    filter_by = filter_by.strip() if filter_by else ""
    
    logger.info(f"Running query on '{collection_name}': q='{query}', query_by='{query_by}', filter='{filter_by}'")
    
    search_params = {
        "q": query if query else "*",
        "query_by": query_by,
        "per_page": per_page,
    }
    
    # Only add filter if it's not empty and looks valid
    if filter_by:
        search_params["filter_by"] = filter_by
        logger.info(f"Filter applied: {filter_by}")
    
    result = client.collections[collection_name].documents.search(search_params)
    logger.info(f"Query returned {result.get('found', 0)} results in {result.get('search_time_ms', 0)}ms")
    return result


def get_campaign_field_name(client, collection: str) -> str:
    schema = client.collections[collection].retrieve()
    field_names = [f["name"] for f in schema.get("fields", [])]
    for candidate in ["campaign_name", "campaign"]:
        if candidate in field_names:
            return candidate
    for name in field_names:
        if "campaign" in name:
            return name
    for f in schema.get("fields", []):
        if f.get("type") == "string":
            return f["name"]
    return "id"


def search_campaigns(client, collection, query, limit=50):
    if not query:
        return [], "campaign"
    query_field = get_campaign_field_name(client, collection)
    logger.info(f"Searching '{query}' in collection '{collection}' (field: {query_field}, limit: {limit})")
    res = client.collections[collection].documents.search({
        "q": query,
        "query_by": query_field,
        "per_page": limit,
    })
    hits = res.get("hits", [])
    logger.info(f"Found {len(hits)} results")
    return [h["document"] for h in hits], query_field


def detect_fields(client, collection: str):
    logger.debug(f"Detecting fields for collection '{collection}'")
    schema = client.collections[collection].retrieve()
    field_names = [f["name"] for f in schema.get("fields", [])]

    def pick(candidates, default=None):
        for c in candidates:
            if c in field_names:
                return c
        for name in field_names:
            if any(c in name for c in candidates):
                return name
        return default

    fields = {
        "campaign": get_campaign_field_name(client, collection),
        "keyword": pick(["customer_search_term", "search_term", "keyword", "term", "targeting"]),
        "impressions": pick(["impressions", "impression"]),
        "clicks": pick(["clicks", "click"]),
        "match_type": pick(["match_type", "matchtype"]),
        "targeting": pick(["targeting", "target"]),
    }
    logger.debug(f"Detected fields: {fields}")
    return fields


def fetch_campaign_docs(client, collection: str, campaign: str, limit: int = 500):
    logger.info(f"Fetching docs for campaign '{campaign}' from '{collection}'")
    fields = detect_fields(client, collection)
    campaign_field = fields["campaign"]
    if not campaign_field:
        raise ValueError(f"Cannot infer campaign field for collection {collection}")
    res = client.collections[collection].documents.search({
        "q": campaign,
        "query_by": campaign_field,
        "per_page": limit,
    })
    docs = [h["document"] for h in res.get("hits", [])]
    logger.info(f"Fetched {len(docs)} documents")
    return docs, fields


def compute_ctr(df: pd.DataFrame, clicks_field: str, impressions_field: str):
    if clicks_field not in df.columns or impressions_field not in df.columns:
        return df
    df = df.copy()
    df["ctr_calc"] = (df[clicks_field].astype(float) / df[impressions_field].replace(0, 1).astype(float)) * 100
    df["ctr_calc"] = df["ctr_calc"].fillna(0)
    return df


def aggregate_keywords(df: pd.DataFrame, fields: dict):
    keyword_field = fields.get("keyword")
    impressions_field = fields.get("impressions")
    clicks_field = fields.get("clicks")
    match_field = fields.get("match_type")
    targeting_field = fields.get("targeting")

    if not keyword_field or keyword_field not in df.columns:
        return df
    if impressions_field not in df.columns or clicks_field not in df.columns:
        return df

    agg_map = {impressions_field: "sum", clicks_field: "sum"}
    
    # Only add match_field if it's different from keyword_field and exists
    if match_field and match_field in df.columns and match_field != keyword_field:
        agg_map[match_field] = lambda x: next((v for v in x if pd.notna(v)), None)
    
    # Only add targeting_field if it's different from keyword_field and match_field, and exists
    if targeting_field and targeting_field in df.columns and targeting_field != keyword_field:
        if targeting_field not in agg_map:  # Avoid duplicate columns
            agg_map[targeting_field] = lambda x: next((v for v in x if pd.notna(v)), None)

    result = df.groupby(keyword_field, as_index=False).agg(agg_map).reset_index(drop=True)
    
    # Ensure no duplicate columns (safety check)
    result = result.loc[:, ~result.columns.duplicated()]
    
    return result


# ==================== MAIN APP ====================

def init_session_state():
    """Initialize session state variables for persisting results across tab switches."""
    if "campaign_results" not in st.session_state:
        st.session_state.campaign_results = None
    if "keyword_results" not in st.session_state:
        st.session_state.keyword_results = None
    if "ai_recommendations" not in st.session_state:
        st.session_state.ai_recommendations = None
    if "ai_output_file" not in st.session_state:
        st.session_state.ai_output_file = None
    if "explorer_query_results" not in st.session_state:
        st.session_state.explorer_query_results = None


def main():
    logger.info("Rendering dashboard...")
    
    # Initialize session state
    init_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 Amazon Ads Campaign Analytics</h1>
        <p>AI-Powered Campaign Optimization • Real-time Search • Keyword Insights</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize Typesense client
    try:
        client = get_client()
        collections = list_collections(client)
        logger.info(f"Connected to Typesense. Collections: {collections}")
    except Exception as e:
        logger.error(f"Failed to connect to Typesense: {e}")
        st.error(f"❌ Failed to connect to Typesense: {e}")
        st.info("Make sure Typesense is running and accessible.")
        return

    if not collections:
        logger.warning("No collections found in Typesense")
        st.warning("⚠️ No collections found in Typesense. Run the ingestion script first.")
        st.code("python typesense_ingest.py --api-key mykey --drop-existing", language="bash")
        return

    # ==================== SIDEBAR ====================
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Collection selector
        st.markdown("**Data Collections**")
        default_collection = "amazon_ads_sponsored_products_campaign_l30"
        default_idx = collections.index(default_collection) if default_collection in collections else 0
        
        selected_collections = st.multiselect(
            "Select collections",
            options=collections,
            default=[collections[default_idx]] if collections else [],
            help="Choose which data sources to search",
        )
        
        st.divider()
        
        # API Settings
        st.markdown("**API Settings**")
        api_key_input = st.text_input(
            "OpenAI API Key",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
            help="Required for AI recommendations",
        )
        model_input = st.selectbox(
            "GPT Model",
            options=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-5-mini"],
            index=0,
        )
        
        st.divider()
        
        # Token Usage Display
        st.markdown("**📊 Token Usage**")
        try:
            tracker = get_tracker()
            totals = tracker.get_totals()
            
            col_token1, col_token2 = st.columns(2)
            with col_token1:
                st.metric("Input Tokens", f"{totals['input_tokens']:,}")
            with col_token2:
                st.metric("Output Tokens", f"{totals['output_tokens']:,}")
            
            st.metric("Total Tokens", f"{totals['total_tokens']:,}")
            
            # Show recent usage
            recent = tracker.get_recent_usage(5)
            if recent:
                with st.expander("📝 Recent Usage", expanded=False):
                    for entry in reversed(recent[-5:]):
                        st.caption(
                            f"**{entry.get('campaign', 'N/A')}** - "
                            f"{entry['input_tokens']:,} in + {entry['output_tokens']:,} out = "
                            f"{entry['total_tokens']:,} total ({entry.get('model', 'N/A')})"
                        )
            
            if st.button("🔄 Reset Totals", help="Reset cumulative token totals"):
                tracker.reset_totals()
                st.rerun()
        except Exception as e:
            st.warning(f"Token tracking unavailable: {e}")
        
        st.divider()
        
        # Analysis Parameters
        st.markdown("**Analysis Parameters**")
        top_n = st.slider("Top N keywords", 5, 100, 20, 5)
        min_impr = st.slider("Min impressions", 0, 1000, 50, 10)
        low_ctr_threshold = st.slider("Low CTR threshold (%)", 0.0, 10.0, 1.0, 0.1)
        limit = st.slider("Max results", 50, 500, 200, 50)

    # ==================== MAIN CONTENT ====================
    
    # Campaign Input - Main focus
    st.markdown('<div class="section-header">🎯 Campaign Selection</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        campaign_name = st.text_input(
            "Enter Campaign Name",
            placeholder="Type campaign name to search...",
            help="Enter the full or partial campaign name",
            label_visibility="collapsed",
        )
    with col2:
        search_btn = st.button("🔍 Search", type="primary", width='stretch')

    if not selected_collections:
        st.info("👈 Select at least one collection from the sidebar to begin.")
        return

    # ==================== TABS FOR DIFFERENT VIEWS ====================
    
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Campaign Data", "🔑 Keyword Analysis", "🤖 AI Recommendations", "🗄️ Typesense Explorer"])

    # ==================== TAB 1: CAMPAIGN DATA ====================
    with tab1:
        st.markdown('<div class="section-header">Campaign Search Results</div>', unsafe_allow_html=True)
        
        # Clear button
        col_clear1, col_clear2 = st.columns([4, 1])
        with col_clear2:
            if st.button("🗑️ Clear", key="clear_campaign", width='stretch'):
                st.session_state.campaign_results = None
                st.rerun()
        
        # Search and store results
        if search_btn and campaign_name:
            logger.info(f"Searching campaigns: {campaign_name}")
            results = []
            for col in selected_collections:
                try:
                    docs, field_used = search_campaigns(client, col, campaign_name, limit=limit)
                    results.append((col, field_used, docs))
                except Exception as e:
                    results.append((col, "error", f"Error: {e}"))
            
            # Store in session state
            st.session_state.campaign_results = {
                "query": campaign_name,
                "results": results,
            }

        # Display results from session state
        if st.session_state.campaign_results:
            results = st.session_state.campaign_results["results"]
            query = st.session_state.campaign_results["query"]
            st.caption(f"Results for: **{query}**")
            
            if results:
                sub_tabs = st.tabs([name.split("_")[-1] for name, _, _ in results])
                for sub_tab, (col, field_used, docs) in zip(sub_tabs, results):
                    with sub_tab:
                        if isinstance(docs, str):
                            st.error(docs)
                        elif not docs:
                            st.warning("No matching campaigns found.")
                        else:
                            st.caption(f"Collection: `{col}` | Query field: `{field_used}`")
                            st.metric("Rows Found", len(docs))
                            st.dataframe(pd.DataFrame(docs), width='stretch')
        else:
            st.info("Enter a campaign name and click **Search** to view campaign data.")

    # ==================== TAB 2: KEYWORD ANALYSIS ====================
    with tab2:
        st.markdown('<div class="section-header">Keyword Performance Analysis</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            keyword_campaign = st.text_input(
                "Campaign for keyword analysis",
                value=campaign_name,
                key="keyword_campaign",
                placeholder="Enter exact campaign name...",
                label_visibility="collapsed",
            )
        with col2:
            keyword_btn = st.button("📊 Analyze", type="secondary", width='stretch')
        with col3:
            if st.button("🗑️ Clear", key="clear_keywords", width='stretch'):
                st.session_state.keyword_results = None
                st.rerun()

        # Analyze and store results
        if keyword_btn and keyword_campaign:
            logger.info(f"Analyzing keywords for: {keyword_campaign}")
            keyword_data = []
            for col_name in selected_collections:
                try:
                    docs, fields = fetch_campaign_docs(client, col_name, keyword_campaign, limit=limit)
                    keyword_field = fields["keyword"]
                    impressions_field = fields["impressions"]
                    clicks_field = fields["clicks"]

                    if not keyword_field or not impressions_field or not clicks_field:
                        keyword_data.append({"collection": col_name, "error": "Missing required fields"})
                        continue

                    df = pd.DataFrame(docs)
                    if df.empty:
                        keyword_data.append({"collection": col_name, "error": "No data found"})
                        continue

                    # Ensure numeric
                    for c in [impressions_field, clicks_field]:
                        if c in df.columns:
                            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

                    df = aggregate_keywords(df, fields)
                    df = compute_ctr(df, clicks_field, impressions_field)
                    df = df[df[impressions_field] >= min_impr]

                    if df.empty:
                        keyword_data.append({"collection": col_name, "error": f"No keywords meet threshold ({min_impr})"})
                        continue

                    # Best keywords
                    best = (
                        df[(df[clicks_field].astype(float) > 0) & (df["ctr_calc"] > low_ctr_threshold)]
                        .sort_values(["ctr_calc", impressions_field], ascending=[False, False])
                        .head(top_n)
                        .reset_index(drop=True)
                    )

                    # Low CTR keywords
                    worst = (
                        df[(df[impressions_field] >= min_impr) & 
                           ((df["ctr_calc"] <= low_ctr_threshold) | (df[clicks_field].astype(float) == 0))]
                        .sort_values([impressions_field, "ctr_calc"], ascending=[False, True])
                        .head(top_n)
                        .reset_index(drop=True)
                    )

                    display_cols = [keyword_field, impressions_field, clicks_field, "ctr_calc"]
                    if fields.get("match_type") and fields["match_type"] in df.columns:
                        if fields["match_type"] not in display_cols:
                            display_cols.insert(1, fields["match_type"])
                    if fields.get("targeting") and fields["targeting"] in df.columns:
                        if fields["targeting"] not in display_cols:
                            display_cols.insert(1, fields["targeting"])
                    
                    # Ensure no duplicate columns
                    display_cols = list(dict.fromkeys(display_cols))

                    keyword_data.append({
                        "collection": col_name,
                        "best": best,
                        "worst": worst,
                        "display_cols": display_cols,
                    })
                except Exception as e:
                    keyword_data.append({"collection": col_name, "error": str(e)})
            
            # Store in session state
            st.session_state.keyword_results = {
                "campaign": keyword_campaign,
                "data": keyword_data,
            }

        # Display results from session state
        if st.session_state.keyword_results:
            results = st.session_state.keyword_results
            st.caption(f"Keyword analysis for: **{results['campaign']}**")
            
            for item in results["data"]:
                with st.expander(f"📁 {item['collection']}", expanded=True):
                    if "error" in item:
                        st.warning(item["error"])
                    else:
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown("**✅ Top Performing Keywords**")
                            display_cols = list(dict.fromkeys(item["display_cols"]))  # Remove duplicates
                            valid_cols = [c for c in display_cols if c in item["best"].columns]
                            best_df = item["best"][valid_cols].copy()
                            best_df = best_df.loc[:, ~best_df.columns.duplicated()]  # Safety
                            st.dataframe(best_df, width='stretch', hide_index=True)
                        with col_b:
                            st.markdown("**⚠️ Underperforming Keywords**")
                            valid_cols = [c for c in display_cols if c in item["worst"].columns]
                            worst_df = item["worst"][valid_cols].copy()
                            worst_df = worst_df.loc[:, ~worst_df.columns.duplicated()]  # Safety
                            st.dataframe(worst_df, width='stretch', hide_index=True)
        else:
            st.info("Enter a campaign name and click **Analyze** to see keyword performance.")

    # ==================== TAB 3: AI RECOMMENDATIONS ====================
    with tab3:
        st.markdown('<div class="section-header">AI-Powered Optimization Recommendations</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            rec_campaign = st.text_input(
                "Campaign for AI analysis",
                value=campaign_name,
                key="rec_campaign",
                placeholder="Enter exact campaign name...",
                label_visibility="collapsed",
            )
        with col2:
            rec_btn = st.button("🤖 Generate", type="primary", width='stretch')
        with col3:
            if st.button("🗑️ Clear", key="clear_ai", width='stretch'):
                st.session_state.ai_recommendations = None
                st.session_state.ai_output_file = None
                st.rerun()

        # Generate and store results
        if rec_btn and rec_campaign:
            api_key = api_key_input or os.getenv("OPENAI_API_KEY")
            if not api_key:
                st.error("❌ OpenAI API key is required. Add it in the sidebar.")
            else:
                try:
                    logger.info(f"Generating AI recommendations for: {rec_campaign}")
                    
                    # Status indicator
                    status = st.status("Generating AI recommendations...", expanded=True)
                    
                    with status:
                        st.write("🔄 Fetching campaign data from Typesense...")
                        st.write("📊 Computing metrics...")
                        st.write("🤖 Streaming Agent  response...")
                    
                    # Streaming response container
                    response_container = st.container()
                    with response_container:
                        response_placeholder = st.empty()
                        full_text = ""
                        output_file = None

                        for chunk in generate_recommendations_streaming(
                            campaign=rec_campaign,
                            ts_host=os.getenv("TYPESENSE_HOST", "localhost"),
                            ts_port=int(os.getenv("TYPESENSE_PORT", "8108")),
                            ts_protocol=os.getenv("TYPESENSE_PROTOCOL", "http"),
                            ts_api_key=os.getenv("TYPESENSE_API_KEY", "mykey"),
                            api_key=api_key,
                            model=model_input or "gpt-5-mini",
                            top_n=top_n,
                            min_impr=min_impr,
                        ):
                            if isinstance(chunk, dict) and chunk.get("complete"):
                                output_file = chunk.get("output_file")
                                full_text = chunk.get("full_text", full_text)
                            else:
                                full_text += chunk
                                response_placeholder.markdown(full_text + "▌")

                        # Final display
                        response_placeholder.markdown(full_text)
                    
                    status.update(label="✅ Recommendations generated!", state="complete")
                    
                    # Get token usage for this request
                    try:
                        tracker = get_tracker()
                        recent_usage = tracker.get_recent_usage(1)
                        if recent_usage:
                            latest = recent_usage[0]
                            st.info(
                                f"📊 **Token Usage (This Request):** "
                                f"{latest['input_tokens']:,} input + {latest['output_tokens']:,} output = "
                                f"{latest['total_tokens']:,} total tokens"
                            )
                    except Exception as e:
                        logger.warning(f"Could not display token usage: {e}")
                    
                    # Store in session state
                    st.session_state.ai_recommendations = {
                        "campaign": rec_campaign,
                        "text": full_text,
                    }
                    st.session_state.ai_output_file = output_file

                    # Success info
                    if output_file:
                        st.success(f"📁 Saved to: `{output_file}`")

                    # Copy section
                    with st.expander("📋 Copy Recommendations", expanded=False):
                        st.code(full_text, language="markdown")

                except FileNotFoundError as e:
                    st.error(f"❌ {e}")
                except Exception as e:
                    st.error(f"❌ Failed to generate recommendations: {e}")
        
        # Display stored results if not generating new ones
        elif st.session_state.ai_recommendations:
            results = st.session_state.ai_recommendations
            st.caption(f"Recommendations for: **{results['campaign']}**")
            
            if st.session_state.ai_output_file:
                st.success(f"📁 Saved to: `{st.session_state.ai_output_file}`")
            
            st.markdown(results["text"])
            
            with st.expander("📋 Copy Recommendations", expanded=False):
                st.code(results["text"], language="markdown")
        
        else:
            st.info("Enter a campaign name and click **Generate** for AI-powered optimization insights.")
            
            # Show what the AI can do
            st.markdown("""
            **What you'll get:**
            - 📈 Budget optimization recommendations
            - 💰 Bid strategy adjustments
            - 🎯 Placement optimization
            - 🔑 Keyword expansion opportunities
            - 🚫 Negative keyword suggestions
            - ⚡ Quick wins for immediate impact
            """)

    # ==================== TAB 4: TYPESENSE EXPLORER ====================
    with tab4:
        st.markdown('<div class="section-header">Typesense Data Explorer</div>', unsafe_allow_html=True)
        
        # Collection Stats Section
        st.markdown("### 📊 Collection Statistics")
        
        try:
            stats = get_collection_stats(client)
            
            if stats:
                # Create metrics cards
                cols = st.columns(len(stats))
                for i, stat in enumerate(stats):
                    with cols[i]:
                        short_name = stat["name"].replace("amazon_ads_", "").replace("_", " ").title()
                        st.metric(
                            label=short_name[:20] + "..." if len(short_name) > 20 else short_name,
                            value=f"{stat['num_documents']:,}",
                            help=f"Collection: {stat['name']}"
                        )
                
                # Detailed stats table
                with st.expander("📋 Detailed Collection Stats", expanded=False):
                    stats_df = pd.DataFrame(stats)
                    stats_df.columns = ["Collection Name", "Documents", "Fields", "Created At"]
                    st.dataframe(stats_df, width='stretch', hide_index=True)
            else:
                st.warning("No collections found.")
        except Exception as e:
            st.error(f"Error fetching stats: {e}")
        
        st.divider()
        
        # Schema Viewer Section
        st.markdown("### 🔍 Schema Viewer")
        
        schema_col = st.selectbox(
            "Select collection to view schema",
            options=collections,
            key="schema_collection",
        )
        
        if schema_col:
            try:
                schema = get_collection_schema(client, schema_col)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Documents", f"{schema.get('num_documents', 0):,}")
                with col2:
                    st.metric("Fields", len(schema.get("fields", [])))
                
                # Fields table
                fields = schema.get("fields", [])
                if fields:
                    fields_data = []
                    for f in fields:
                        fields_data.append({
                            "Field Name": f.get("name", ""),
                            "Type": f.get("type", ""),
                            "Facet": "✓" if f.get("facet") else "✗",
                            "Index": "✓" if f.get("index", True) else "✗",
                            "Optional": "✓" if f.get("optional") else "✗",
                        })
                    st.dataframe(pd.DataFrame(fields_data), width='stretch', hide_index=True)
                
                # Raw schema JSON
                with st.expander("📄 Raw Schema JSON", expanded=False):
                    st.json(schema)
                    
            except Exception as e:
                st.error(f"Error fetching schema: {e}")
        
        st.divider()
        
        # Query Runner Section
        st.markdown("### 🔎 Query Runner")
        
        query_col = st.selectbox(
            "Select collection to query",
            options=collections,
            key="query_collection",
        )
        
        if query_col:
            # Get available fields for query_by
            try:
                schema = get_collection_schema(client, query_col)
                string_fields = [f["name"] for f in schema.get("fields", []) 
                                if f.get("type") in ["string", "string[]"]]
                all_fields = [f["name"] for f in schema.get("fields", [])]
            except:
                string_fields = ["id"]
                all_fields = ["id"]
            
            col1, col2 = st.columns(2)
            with col1:
                query_text = st.text_input(
                    "Search query",
                    value="*",
                    placeholder="Enter search term or * for all",
                    key="explorer_query_text",
                )
            with col2:
                query_by = st.selectbox(
                    "Query by field",
                    options=string_fields if string_fields else ["id"],
                    key="explorer_query_by",
                )
            
            col3, col4 = st.columns(2)
            with col3:
                filter_by = st.text_input(
                    "Filter (optional)",
                    placeholder="Leave empty for no filter",
                    key="explorer_filter",
                )
            with col4:
                per_page = st.number_input(
                    "Results per page",
                    min_value=1,
                    max_value=250,
                    value=20,
                    key="explorer_per_page",
                )
            
            # Filter syntax help
            with st.expander("💡 Filter Syntax Help", expanded=False):
                st.markdown("""
                **Numeric filters:**
                - `impressions:>100` — greater than
                - `clicks:>=10` — greater or equal
                - `spend:<50` — less than
                - `impressions:[100..500]` — range (inclusive)
                
                **String filters:**
                - `campaign_name:=MyCampaign` — exact match
                - `match_type:=EXACT` — exact match
                
                **Multiple filters:**
                - `impressions:>100 && clicks:>0` — AND
                - `match_type:=EXACT || match_type:=PHRASE` — OR
                """)
            
            col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
            with col_btn1:
                run_query_btn = st.button("▶️ Run Query", type="primary", width='stretch')
            with col_btn2:
                export_btn = st.button("📥 Export All", width='stretch', help="Export entire collection")
            with col_btn3:
                if st.button("🗑️ Clear", key="clear_explorer", width='stretch'):
                    st.session_state.explorer_query_results = None
                    st.rerun()
            
            # Run query
            if run_query_btn:
                try:
                    logger.info(f"Running explorer query: collection={query_col}, q={query_text}, query_by={query_by}")
                    result = run_typesense_query(
                        client, 
                        query_col, 
                        query_text, 
                        query_by, 
                        filter_by, 
                        per_page
                    )
                    
                    st.session_state.explorer_query_results = {
                        "collection": query_col,
                        "query": query_text,
                        "result": result,
                    }
                except typesense.exceptions.RequestMalformed as e:
                    st.error(f"❌ Invalid filter syntax: {e}")
                    st.info("💡 Check the **Filter Syntax Help** above for valid filter examples.")
                except Exception as e:
                    error_msg = str(e)
                    if "filter" in error_msg.lower():
                        st.error(f"❌ Filter error: {error_msg}")
                        st.info("💡 Leave the filter field empty if you don't need filtering, or check the syntax help above.")
                    else:
                        st.error(f"❌ Query error: {error_msg}")
            
            # Export collection
            if export_btn:
                try:
                    with st.spinner("Exporting collection..."):
                        export_str = client.collections[query_col].documents.export()
                        if export_str:
                            import json
                            lines = [json.loads(line) for line in export_str.splitlines() if line.strip()]
                            export_df = pd.DataFrame(lines)
                            
                            st.success(f"Exported {len(export_df):,} documents")
                            st.download_button(
                                "📥 Download CSV",
                                data=export_df.to_csv(index=False),
                                file_name=f"{query_col}_export.csv",
                                mime="text/csv",
                            )
                        else:
                            st.warning("Collection is empty")
                except Exception as e:
                    st.error(f"Export error: {e}")
            
            # Display results
            if st.session_state.explorer_query_results:
                results = st.session_state.explorer_query_results
                result = results["result"]
                
                st.markdown(f"**Results for:** `{results['query']}` in `{results['collection']}`")
                
                # Stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Found", f"{result.get('found', 0):,}")
                with col2:
                    st.metric("Returned", len(result.get("hits", [])))
                with col3:
                    st.metric("Search Time", f"{result.get('search_time_ms', 0)} ms")
                
                # Results table
                hits = result.get("hits", [])
                if hits:
                    docs = [h["document"] for h in hits]
                    result_df = pd.DataFrame(docs)
                    st.dataframe(result_df, width='stretch', hide_index=True)
                    
                    # Raw JSON view
                    with st.expander("📄 Raw JSON Results", expanded=False):
                        st.json(result)
                else:
                    st.info("No documents found matching your query.")


if __name__ == "__main__":
    main()
