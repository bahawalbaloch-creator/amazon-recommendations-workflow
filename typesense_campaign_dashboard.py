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
    if match_field and match_field in df.columns:
        agg_map[match_field] = lambda x: next((v for v in x if pd.notna(v)), None)
    if targeting_field and targeting_field in df.columns:
        agg_map[targeting_field] = lambda x: next((v for v in x if pd.notna(v)), None)

    return df.groupby(keyword_field, as_index=False).agg(agg_map).reset_index(drop=True)


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
            options=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
            index=0,
        )
        
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
        search_btn = st.button("🔍 Search", type="primary", use_container_width=True)

    if not selected_collections:
        st.info("👈 Select at least one collection from the sidebar to begin.")
        return

    # ==================== TABS FOR DIFFERENT VIEWS ====================
    
    tab1, tab2, tab3 = st.tabs(["📋 Campaign Data", "🔑 Keyword Analysis", "🤖 AI Recommendations"])

    # ==================== TAB 1: CAMPAIGN DATA ====================
    with tab1:
        st.markdown('<div class="section-header">Campaign Search Results</div>', unsafe_allow_html=True)
        
        # Clear button
        col_clear1, col_clear2 = st.columns([4, 1])
        with col_clear2:
            if st.button("🗑️ Clear", key="clear_campaign", use_container_width=True):
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
                            st.dataframe(pd.DataFrame(docs), use_container_width=True)
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
            keyword_btn = st.button("📊 Analyze", type="secondary", use_container_width=True)
        with col3:
            if st.button("🗑️ Clear", key="clear_keywords", use_container_width=True):
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
                        display_cols.insert(1, fields["match_type"])
                    if fields.get("targeting") and fields["targeting"] in df.columns:
                        display_cols.insert(1, fields["targeting"])

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
                            display_cols = item["display_cols"]
                            st.dataframe(
                                item["best"][[c for c in display_cols if c in item["best"].columns]],
                                use_container_width=True,
                                hide_index=True,
                            )
                        with col_b:
                            st.markdown("**⚠️ Underperforming Keywords**")
                            st.dataframe(
                                item["worst"][[c for c in display_cols if c in item["worst"].columns]],
                                use_container_width=True,
                                hide_index=True,
                            )
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
            rec_btn = st.button("🤖 Generate", type="primary", use_container_width=True)
        with col3:
            if st.button("🗑️ Clear", key="clear_ai", use_container_width=True):
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
                        st.write("🤖 Streaming GPT response...")
                    
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
                            model=model_input or "gpt-4o-mini",
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


if __name__ == "__main__":
    main()
