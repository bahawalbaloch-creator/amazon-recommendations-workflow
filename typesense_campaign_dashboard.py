"""
Streamlit dashboard to look up campaigns stored in Typesense.

Assumes you've ingested campaigns into a Typesense collection, e.g.:
  - amazon_ads_sponsored_products_campaign_l30

Run:
  streamlit run typesense_campaign_dashboard.py
"""

import os

import pandas as pd
import streamlit as st
import typesense
from campaign_recommendations import generate_recommendations, generate_recommendations_streaming


@st.cache_resource
def get_client():
    return typesense.Client(
        {
            "nodes": [
                {
                    "host": os.getenv("TYPESENSE_HOST", "localhost"),
                    "port": int(os.getenv("TYPESENSE_PORT", "8108")),
                    "protocol": os.getenv("TYPESENSE_PROTOCOL", "http"),
                }
            ],
            "api_key": os.getenv("TYPESENSE_API_KEY", "mykey"),
            "connection_timeout_seconds": 5,
        }
    )


def list_collections(client):
    cols = client.collections.retrieve()
    # Show only collections that look like campaign data by default
    return [c["name"] for c in cols]


def get_campaign_field_name(client, collection: str) -> str:
    """Try to detect the best field to search campaigns on for this collection."""
    schema = client.collections[collection].retrieve()
    field_names = [f["name"] for f in schema.get("fields", [])]

    # Prefer common campaign field variants
    for candidate in ["campaign_name", "campaign"]:
        if candidate in field_names:
            return candidate

    # Fallback: first field containing 'campaign'
    for name in field_names:
        if "campaign" in name:
            return name

    # Last resort: just use the first string field
    for f in schema.get("fields", []):
        if f.get("type") == "string":
            return f["name"]

    # If everything fails, use 'id'
    return "id"


def search_campaigns(client, collection, query, limit=50):
    """Search a single collection for the campaign string and return docs + field used."""
    if not query:
        return [], "campaign"
    query_field = get_campaign_field_name(client, collection)
    res = client.collections[collection].documents.search(
        {
            "q": query,
            "query_by": query_field,
            "per_page": limit,
        }
    )
    return [h["document"] for h in res.get("hits", [])], query_field


def detect_fields(client, collection: str):
    """Infer key fields used for keyword analysis."""
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

    campaign_field = get_campaign_field_name(client, collection)
    keyword_field = pick(["customer_search_term", "search_term", "keyword", "term", "targeting"], default=None)
    impressions_field = pick(["impressions", "impression"], default=None)
    clicks_field = pick(["clicks", "click"], default=None)
    match_field = pick(["match_type", "matchtype", "matchtype_raw"], default=None)
    targeting_field = pick(["targeting", "target"], default=None)

    return {
        "campaign": campaign_field,
        "keyword": keyword_field,
        "impressions": impressions_field,
        "clicks": clicks_field,
        "match_type": match_field,
        "targeting": targeting_field,
    }


def fetch_campaign_docs(client, collection: str, campaign: str, limit: int = 500):
    """Fetch documents for a campaign from a collection, returning docs + field info."""
    fields = detect_fields(client, collection)
    campaign_field = fields["campaign"]
    if not campaign_field:
        raise ValueError(f"Cannot infer campaign field for collection {collection}")

    res = client.collections[collection].documents.search(
        {
            "q": campaign,
            "query_by": campaign_field,
            "per_page": limit,
        }
    )
    docs = [h["document"] for h in res.get("hits", [])]
    return docs, fields


def compute_ctr(df: pd.DataFrame, clicks_field: str, impressions_field: str):
    if clicks_field not in df.columns or impressions_field not in df.columns:
        return df
    df = df.copy()
    df["ctr_calc"] = (df[clicks_field].astype(float) / df[impressions_field].replace(0, 1).astype(float)) * 100
    df["ctr_calc"] = df["ctr_calc"].fillna(0)
    return df


def aggregate_keywords(df: pd.DataFrame, fields: dict):
    """Group by keyword to show unique terms with aggregated stats."""
    keyword_field = fields.get("keyword")
    impressions_field = fields.get("impressions")
    clicks_field = fields.get("clicks")
    match_field = fields.get("match_type")
    targeting_field = fields.get("targeting")

    if not keyword_field or keyword_field not in df.columns:
        return df
    if impressions_field not in df.columns or clicks_field not in df.columns:
        return df

    agg_map = {
        impressions_field: "sum",
        clicks_field: "sum",
    }
    # Keep a representative match_type/targeting value (first non-null)
    if match_field and match_field in df.columns:
        agg_map[match_field] = lambda x: next((v for v in x if pd.notna(v)), None)
    if targeting_field and targeting_field in df.columns:
        agg_map[targeting_field] = lambda x: next((v for v in x if pd.notna(v)), None)

    grouped = (
        df.groupby(keyword_field, as_index=False)
        .agg(agg_map)
        .reset_index(drop=True)
    )
    return grouped


def main():
    st.set_page_config(page_title="Campaign Lookup (Typesense)", layout="wide")
    st.title("Campaign Lookup using Typesense")

    st.sidebar.header("Typesense Settings")
    st.sidebar.write("Make sure Typesense is running and data is ingested.")

    client = get_client()
    collections = list_collections(client)

    if not collections:
        st.error("No collections found in Typesense. Run the ingestion script first.")
        return

    default_collection = "amazon_ads_sponsored_products_campaign_l30"
    default_index = max(collections.index(default_collection), 0) if default_collection in collections else 0

    # Make the dropdown show full collection names (no ellipsis)
    st.sidebar.markdown(
        """
        <style>
        /* Widen dropdown and allow text to wrap/overflow visibly */
        div[data-baseweb="select"] span {
            overflow: visible !important;
            text-overflow: initial !important;
            white-space: normal !important;
        }
        div[data-baseweb="select"] {
            max-width: 100% !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    selected_collections = st.sidebar.multiselect(
        "Collections to search",
        options=collections,
        default=[collections[default_index]] if collections else [],
        placeholder="Select one or more collections",
    )

    query = st.text_input("Campaign name contains", "")
    limit = st.slider("Max results per collection", min_value=10, max_value=500, value=200, step=10)

    st.markdown("---")
    st.subheader("Keyword performance for a campaign (best / low CTR)")
    campaign_exact = st.text_input("Campaign name (exact match preferred)", value=query)
    top_n = st.slider("Top N rows to show", min_value=5, max_value=100, value=20, step=5)
    min_impr = st.slider("Min impressions to consider", min_value=0, max_value=1000, value=50, step=10)
    low_ctr_threshold = st.slider("Low CTR threshold (%)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    api_key_input = st.text_input("OpenAI API key (optional, uses OPENAI_API_KEY if blank)", value=os.getenv("OPENAI_API_KEY", ""))
    model_input = st.text_input("OpenAI model", value="gpt-4o-mini")
    data_dir_input = st.text_input("Reports data directory", value=os.getenv("DATA_DIR", "./data"))

    if st.button("Search across collections") and query:
        if not selected_collections:
            st.warning("Select at least one collection.")
            return

        results = []
        for col in selected_collections:
            try:
                docs, field_used = search_campaigns(client, col, query, limit=limit)
                results.append((col, field_used, docs))
            except Exception as e:
                results.append((col, "error", f"Error: {e}"))

        tabs = st.tabs([name for name, _, _ in results])
        for tab, (col, field_used, docs) in zip(tabs, results):
            with tab:
                if isinstance(docs, str):
                    st.error(docs)
                    continue
                st.caption(f"Query field: `{field_used}`")
                if not docs:
                    st.warning("No campaigns found.")
                else:
                    df = pd.DataFrame(docs)
                    st.write(f"Found {len(df)} rows")
                    st.dataframe(df)

    if st.button("Get keyword insights") and campaign_exact:
        if not selected_collections:
            st.warning("Select at least one collection.")
            return

        tabs = st.tabs(selected_collections)
        for tab, col in zip(tabs, selected_collections):
            with tab:
                try:
                    docs, fields = fetch_campaign_docs(client, col, campaign_exact, limit=limit)
                except Exception as e:
                    st.error(f"Error fetching from {col}: {e}")
                    continue

                keyword_field = fields["keyword"]
                impressions_field = fields["impressions"]
                clicks_field = fields["clicks"]

                if not keyword_field or not impressions_field or not clicks_field:
                    st.warning(
                        f"Missing required fields in {col}. "
                        f"Need keyword (~search_term/keyword), impressions, clicks."
                    )
                    continue

                df = pd.DataFrame(docs)
                if df.empty:
                    st.warning("No rows found for this campaign.")
                    continue

                # Ensure numeric for metrics
                for col in [impressions_field, clicks_field]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

                # Aggregate by keyword to show unique terms
                df = aggregate_keywords(df, fields)

                df = compute_ctr(df, clicks_field, impressions_field)
                df = df[df[impressions_field] >= min_impr]

                if df.empty:
                    st.warning(f"No rows meet impression threshold ({min_impr}).")
                    continue

                # Best: require clicks and CTR above threshold, then rank by CTR desc, impressions desc
                best = (
                    df[
                        (df[clicks_field].astype(float) > 0)
                        & (df["ctr_calc"] > low_ctr_threshold)
                    ]
                    .sort_values(["ctr_calc", impressions_field], ascending=[False, False])
                    .head(top_n)
                    .reset_index(drop=True)
                )

                # Low CTR: high impressions but CTR at/below threshold (or zero clicks)
                worst = (
                    df[
                        (df[impressions_field] >= min_impr)
                        & ((df["ctr_calc"] <= low_ctr_threshold) | (df[clicks_field].astype(float) == 0))
                    ]
                    .sort_values([impressions_field, "ctr_calc"], ascending=[False, True])
                    .head(top_n)
                    .reset_index(drop=True)
                )

                st.caption(
                    f"Fields: keyword=`{keyword_field}`, impressions=`{impressions_field}`, "
                    f"clicks=`{clicks_field}`, campaign=`{fields['campaign']}`, "
                    f"match_type=`{fields.get('match_type')}`, targeting=`{fields.get('targeting')}`"
                )

                display_cols = [keyword_field, impressions_field, clicks_field, "ctr_calc"]
                if fields.get("match_type") and fields["match_type"] in df.columns:
                    display_cols.insert(1, fields["match_type"])
                if fields.get("targeting") and fields["targeting"] in df.columns:
                    display_cols.insert(1, fields["targeting"])

                st.write("**Best keywords (high impressions & CTR)**")
                st.dataframe(best[[c for c in display_cols if c in best.columns]])

                st.write("**Low CTR keywords (high impressions, low CTR)**")
                st.dataframe(worst[[c for c in display_cols if c in worst.columns]])

    # --- GPT recommendations ---
    if st.button("Generate GPT recommendations") and campaign_exact:
        api_key = api_key_input or os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("Missing OpenAI API key.")
        else:
            try:
                st.subheader("GPT Recommendations")
                st.info("🔄 Streaming response...")
                
                # Create a placeholder for streaming text
                response_placeholder = st.empty()
                full_text = ""
                output_file = None
                
                # Stream the recommendations
                for chunk in generate_recommendations_streaming(
                    campaign=campaign_exact,
                    ts_host=os.getenv("TYPESENSE_HOST", "localhost"),
                    ts_port=int(os.getenv("TYPESENSE_PORT", "8108")),
                    ts_protocol=os.getenv("TYPESENSE_PROTOCOL", "http"),
                    ts_api_key=os.getenv("TYPESENSE_API_KEY", "mykey"),
                    api_key=api_key,
                    model=model_input or "gpt-4o-mini",
                    top_n=top_n,
                    min_impr=min_impr,
                ):
                    # Check if this is the final metadata chunk
                    if isinstance(chunk, dict) and chunk.get("complete"):
                        output_file = chunk.get("output_file")
                        full_text = chunk.get("full_text", full_text)
                    else:
                        # Accumulate text and update display
                        full_text += chunk
                        # Display as markdown for proper formatting during streaming
                        response_placeholder.markdown(full_text + "▌")
                
                # Final display without cursor
                response_placeholder.markdown(full_text)
                
                # Show success message with file path
                if output_file:
                    st.success(f"✅ Saved to: {output_file}")
                
                # Copy button using code block
                st.divider()
                st.caption("📋 Click the copy icon (top-right) to copy recommendations:")
                st.code(full_text, language="markdown")
                
            except FileNotFoundError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Failed to generate recommendations: {e}")


if __name__ == "__main__":
    main()

