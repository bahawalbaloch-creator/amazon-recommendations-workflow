"""
Generate descriptive recommendations for a single campaign.

Steps:
1) Load & process reports (same preprocessing as pipeline).
2) Compute campaign KPIs and keyword performance (CTR/CPC/ACOS/ROAS) upfront.
3) Identify best and worst keywords by CTR (with impression thresholds).
4) Include budget utilization metrics.
5) Call OpenAI chat completion with the precomputed metrics for narrative guidance.

Usage:
  python campaign_recommendations.py --campaign "My Campaign" --api-key sk-xxx \
      --data-dir ./data --top-n 10 --min-impr 50
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import openai
import pandas as pd
import typesense
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def find_keyword_column(df: pd.DataFrame) -> str:
    """Find the keyword/search term column name in the dataframe."""
    for candidate in ["customer_search_term", "search_term", "keyword", "term", "targeting"]:
        if candidate in df.columns:
            return candidate
    for col in df.columns:
        if "search" in col.lower() or "keyword" in col.lower() or "term" in col.lower():
            return col
    raise KeyError("No keyword/search term column found in dataframe")


def find_metric_column(df: pd.DataFrame, metric: str) -> str:
    """Find the actual column name for a metric (handles snake_case variations)."""
    # Direct match
    if metric in df.columns:
        return metric
    
    # Common variations (including trailing underscores from snake_case)
    variations = {
        "sales": ["sales", "7_day_total_sales", "7_day_total_sales_", "total_sales", "revenue"],
        "spend": ["spend", "cost", "total_spend"],
        "clicks": ["clicks", "click", "total_clicks"],
        "impressions": ["impressions", "impression", "impr", "total_impressions"],
        "budget_amount": ["budget_amount", "budget", "daily_budget", "budget_amount_"],
    }
    
    candidates = variations.get(metric, [metric])
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    
    # Fuzzy match: look for column containing the metric name
    for col in df.columns:
        if metric in col.lower():
            return col
    
    return None


def clean_currency(val):
    """Strip currency symbols and convert to numeric."""
    if pd.isna(val):
        return 0
    if isinstance(val, (int, float)):
        return val
    # Remove $, commas, and whitespace
    cleaned = str(val).replace("$", "").replace(",", "").strip()
    try:
        return float(cleaned) if cleaned else 0
    except ValueError:
        return 0


def normalize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize metric column names to standard names and ensure numeric."""
    df = df.copy()
    
    # Map actual columns to standard names
    mappings = {
        "sales": find_metric_column(df, "sales"),
        "spend": find_metric_column(df, "spend"),
        "clicks": find_metric_column(df, "clicks"),
        "impressions": find_metric_column(df, "impressions"),
        "budget_amount": find_metric_column(df, "budget_amount"),
    }
    
    # Rename columns to standard names
    for std_name, actual_name in mappings.items():
        if actual_name and actual_name != std_name and actual_name in df.columns:
            df[std_name] = df[actual_name]
    
    # Ensure numeric (with currency symbol stripping for money columns)
    money_cols = ["sales", "spend", "budget_amount"]
    for col in money_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_currency)
    
    # Regular numeric columns
    for col in ["clicks", "impressions"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    
    return df


def ensure_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Convert columns to numeric, filling NaN with 0."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df


def compute_keyword_metrics(df: pd.DataFrame, min_impr: int, top_n: int):
    """Aggregate by search term and compute CTR/CPC/ACOS/ROAS, then pick best/worst by CTR."""
    keyword_col = find_keyword_column(df)

    # Normalize metric columns (handles 7_day_total_sales -> sales, etc.)
    df = normalize_metrics(df)

    agg = (
        df.groupby(keyword_col, as_index=False)
        .agg(
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
            spend=("spend", "sum"),
            sales=("sales", "sum"),
        )
    )

    agg["ctr"] = (agg["clicks"] / agg["impressions"].replace(0, 1)) * 100
    agg["cpc"] = agg["spend"] / agg["clicks"].replace(0, 1)
    agg["acos"] = (agg["spend"] / agg["sales"].replace(0, 1)) * 100
    agg["roas"] = agg["sales"] / agg["spend"].replace(0, 1)
    agg = agg.replace([float("inf"), float("-inf")], 0)

    filtered = agg[agg["impressions"] >= min_impr].copy()

    best = (
        filtered.sort_values(["ctr", "impressions"], ascending=[False, False])
        .head(top_n)
        .to_dict("records")
    )
    worst = (
        filtered.sort_values(["ctr", "impressions"], ascending=[True, False])
        .head(top_n)
        .to_dict("records")
    )

    return agg.to_dict("records"), best, worst


def build_prompt(data: Dict[str, Any]) -> str:
    """Build a detailed, quantitative prompt for GPT with precomputed metrics."""
    return (
        "You are an Amazon Sponsored Products optimization expert with deep expertise in PPC analytics.\n\n"
        
        "# YOUR TASK\n"
        "Analyze the campaign data below and provide 8-12 specific, quantitative recommendations.\n"
        "Each recommendation must include:\n"
        "- Exact numbers (current vs. target metrics)\n"
        "- Expected impact (e.g., 'reduce wasted spend by $X' or 'increase CTR from X% to Y%')\n"
        "- Specific action steps with thresholds\n"
        "- Priority level (High/Medium/Low) based on potential ROI impact\n\n"
        
        "# CAMPAIGN DATA\n"
        f"Campaign Summary:\n{json.dumps(data['campaign_summary'], indent=2, ensure_ascii=False)}\n\n"
        
        f"Budget Metrics:\n{json.dumps(data['budget'], indent=2, ensure_ascii=False)}\n\n"
        
        f"Top Performing Keywords (High CTR):\n{json.dumps(data['best_keywords'], indent=2, ensure_ascii=False)}\n\n"
        
        f"Underperforming Keywords (Low CTR):\n{json.dumps(data['worst_keywords'], indent=2, ensure_ascii=False)}\n\n"
        
        f"Complete Keyword Performance Rollup:\n{json.dumps(data['keyword_rollup'], indent=2, ensure_ascii=False)[:5000]}\n\n"
        
        "# ANALYSIS FRAMEWORK\n"
        "Structure your recommendations across these categories:\n\n"
        
        "## 1. BUDGET OPTIMIZATION\n"
        "- Calculate budget utilization rate and daily pacing\n"
        "- Identify if campaign is budget-constrained (>90% spend) or under-utilizing (<70% spend)\n"
        "- Recommend specific budget adjustments with dollar amounts and rationale\n"
        "- Project impact on impression share and potential sales\n\n"
        
        "## 2. BID STRATEGY\n"
        "- Analyze current avg CPC vs. category benchmarks\n"
        "- Calculate bid efficiency: (Conversions × AOV) / Cost\n"
        "- Recommend specific bid increases/decreases by keyword or ad group (e.g., 'increase bid by $0.15 for keyword X')\n"
        "- Identify keywords with high CVR but low impression share (bid too low)\n\n"
        
        "## 3. PLACEMENT OPTIMIZATION\n"
        "- Compare performance across Top of Search, Product Pages, and Rest of Search\n"
        "- Calculate placement-specific ACOS and conversion rates\n"
        "- Recommend bid multiplier adjustments (e.g., 'increase Top of Search multiplier from X% to Y%')\n"
        "- Quantify expected cost and revenue impact\n\n"
        
        "## 4. KEYWORD EXPANSION\n"
        "- Identify top 3-5 keywords to scale (high CTR, high CVR, ACOS below target)\n"
        "- Recommend specific actions: increase bids by $X, allocate $Y additional daily budget\n"
        "- Suggest match type optimization (e.g., 'add exact match variant for keyword X')\n"
        "- Project incremental sales potential\n\n"
        
        "## 5. NEGATIVE KEYWORD STRATEGY\n"
        "- List specific keywords to pause (e.g., 'CTR < 0.3%, spend > $50, 0 conversions')\n"
        "- Calculate total wasted spend to be eliminated\n"
        "- Recommend negative keyword additions to prevent irrelevant traffic\n"
        "- Estimate monthly savings\n\n"
        
        "## 6. PERFORMANCE ANOMALIES\n"
        "- Flag keywords with unusual metrics (e.g., 'CTR 3x campaign average but 0 conversions')\n"
        "- Identify potential listing issues, pricing problems, or search term mismatches\n"
        "- Suggest diagnostic actions\n\n"
        
        "## 7. QUICK WINS\n"
        "- Prioritize 2-3 immediate actions that can be implemented in <15 minutes\n"
        "- Each quick win must have clear success metrics\n"
        "- Estimate time to impact and expected improvement\n\n"
        
        "# OUTPUT FORMAT\n"
        "For each recommendation, use this structure:\n"
        "**[Priority] Category: Specific Action**\n"
        "- Current State: [exact metrics]\n"
        "- Recommended Action: [specific change with numbers]\n"
        "- Expected Impact: [quantified outcome]\n"
        "- Implementation: [specific steps]\n\n"
        
        "Example:\n"
        "**[HIGH] Budget: Increase Daily Budget**\n"
        "- Current State: $50/day budget, 98% utilization, losing impression share after 2pm\n"
        "- Recommended Action: Increase budget to $75/day (+50%)\n"
        "- Expected Impact: Capture additional 200-300 impressions/day, estimated 3-5 additional conversions/week worth $150-250\n"
        "- Implementation: Adjust in Campaign Settings > Daily Budget, monitor for 7 days\n\n"
        
        "Focus on actionable insights with clear ROI, not generic observations."
    )


def main():
    parser = argparse.ArgumentParser(description="Generate campaign recommendations with GPT using precomputed metrics.")
    parser.add_argument("--campaign", required=True, help="Campaign name to analyze (case-insensitive match).")
    parser.add_argument("--typesense-host", default=os.getenv("TYPESENSE_HOST", "localhost"))
    parser.add_argument("--typesense-port", type=int, default=int(os.getenv("TYPESENSE_PORT", "8108")))
    parser.add_argument("--typesense-protocol", default=os.getenv("TYPESENSE_PROTOCOL", "http"))
    parser.add_argument("--typesense-api-key", default=os.getenv("TYPESENSE_API_KEY"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key.")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI chat model.")
    parser.add_argument("--top-n", type=int, default=10, help="Top N best/worst keywords to include.")
    parser.add_argument("--min-impr", type=int, default=50, help="Minimum impressions to consider a keyword.")
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Missing OpenAI API key. Use --api-key or set OPENAI_API_KEY.")

    text, output_file = generate_recommendations(
        campaign=args.campaign,
        ts_host=args.typesense_host,
        ts_port=args.typesense_port,
        ts_protocol=args.typesense_protocol,
        ts_api_key=args.typesense_api_key,
        api_key=args.api_key,
        model=args.model,
        top_n=args.top_n,
        min_impr=args.min_impr,
    )
    print("\n=== Recommendations ===\n")
    print(text)
    print(f"\n✅ Saved to: {output_file}")


def find_collection(client: typesense.Client, pattern: str) -> str:
    cols = client.collections.retrieve()
    for c in cols:
        if pattern in c["name"]:
            return c["name"]
    raise FileNotFoundError(f"Missing collection containing '{pattern}' in Typesense")


def fetch_collection_df(client: typesense.Client, name: str) -> pd.DataFrame:
    export_str = client.collections[name].documents.export()
    if not export_str:
        return pd.DataFrame()
    lines = [json.loads(line) for line in export_str.splitlines() if line.strip()]
    return pd.DataFrame(lines)


def find_campaign_column(df: pd.DataFrame) -> str:
    """Find the campaign column name in the dataframe (could be 'campaign' or 'campaign_name')."""
    for candidate in ["campaign_name", "campaign"]:
        if candidate in df.columns:
            return candidate
    for col in df.columns:
        if "campaign" in col.lower():
            return col
    raise KeyError("No campaign column found in dataframe")


def generate_recommendations(
    campaign: str,
    ts_host: str,
    ts_port: int,
    ts_protocol: str,
    ts_api_key: str,
    api_key: str,
    model: str = "gpt-4o-mini",
    top_n: int = 10,
    min_impr: int = 50,
) -> str:
    """Compute metrics from Typesense collections and return GPT recommendations as text."""
    if not api_key:
        raise ValueError("Missing OpenAI API key.")
    if not ts_api_key:
        raise ValueError("Missing Typesense API key.")

    client = typesense.Client(
        {
            "nodes": [{"host": ts_host, "port": ts_port, "protocol": ts_protocol}],
            "api_key": ts_api_key,
            "connection_timeout_seconds": 5,
        }
    )

    collections_map = {
        "search_terms": find_collection(client, "search_term"),
        "campaigns": find_collection(client, "campaign"),
        "targeting": find_collection(client, "targeting"),
        "placement": find_collection(client, "placement"),
        "budgets": find_collection(client, "budget"),
    }

    reports = {k: fetch_collection_df(client, v) for k, v in collections_map.items()}

    # Step 2: Campaign filter
    campaign_name = campaign.lower()
    search_terms = reports["search_terms"]
    campaigns_df = reports["campaigns"]
    budgets = reports.get("budgets", pd.DataFrame())

    # Find the campaign column in each dataframe (could be 'campaign' or 'campaign_name')
    st_camp_col = find_campaign_column(search_terms)
    camp_camp_col = find_campaign_column(campaigns_df)

    search_terms = search_terms[search_terms[st_camp_col].str.lower() == campaign_name]
    campaigns_df = campaigns_df[campaigns_df[camp_camp_col].str.lower() == campaign_name]

    if budgets is not None and not budgets.empty:
        try:
            budget_camp_col = find_campaign_column(budgets)
            budgets = budgets[budgets[budget_camp_col].str.lower() == campaign_name]
        except KeyError:
            budgets = pd.DataFrame()

    if campaigns_df.empty:
        raise ValueError(f"Campaign not found: {campaign}")

    # Normalize metric columns (handles 7_day_total_sales -> sales, etc.)
    campaigns_df = normalize_metrics(campaigns_df)

    # Step 3: Campaign summary + budget
    campaign_summary = campaigns_df.groupby(camp_camp_col, as_index=False).agg(
        spend=("spend", "sum"),
        sales=("sales", "sum"),
        clicks=("clicks", "sum"),
        impressions=("impressions", "sum"),
    )
    if "budget_amount" in campaigns_df.columns:
        campaign_summary["budget_amount"] = campaigns_df["budget_amount"].fillna(0).iloc[0] if len(campaigns_df) > 0 else 0
    campaign_summary["ctr"] = (campaign_summary["clicks"] / campaign_summary["impressions"].replace(0, 1)) * 100
    campaign_summary["cvr"] = campaign_summary["sales"] / campaign_summary["clicks"].replace(0, 1)
    campaign_summary["acos"] = (campaign_summary["spend"] / campaign_summary["sales"].replace(0, 1)) * 100
    campaign_summary["roas"] = campaign_summary["sales"] / campaign_summary["spend"].replace(0, 1)
    campaign_summary = campaign_summary.replace([float("inf"), float("-inf")], 0)

    budget_info = {}
    if budgets is not None and not budgets.empty:
        budgets = normalize_metrics(budgets)
        # Debug: print available columns and sample data
        print(f"[DEBUG] Budget columns: {list(budgets.columns)}")
        if len(budgets) > 0:
            print(f"[DEBUG] Budget sample row: {budgets.iloc[0].to_dict()}")
        
        budget_amt = 0
        if "budget_amount" in budgets.columns:
            budget_amt = float(budgets["budget_amount"].sum())
        
        budget_info = {
            "total_budget": budget_amt,
            "daily_budget": budget_amt / max(len(budgets), 1) if budget_amt > 0 else 0,
            "rows": budgets.head(5).to_dict("records"),  # Limit rows to avoid huge payload
        }
        print(f"[DEBUG] Budget info: total={budget_amt}")
    else:
        budget_amt = float(campaign_summary["budget_amount"].iloc[0]) if "budget_amount" in campaign_summary.columns else 0
        budget_info = {"total_budget": budget_amt, "daily_budget": budget_amt}
        print(f"[DEBUG] No budget rows found, using campaign summary budget: {budget_amt}")

    # Step 4: Keywords best/worst
    keyword_rollup, best_keywords, worst_keywords = compute_keyword_metrics(
        search_terms, min_impr=min_impr, top_n=top_n
    )

    # Step 5: Build prompt and call GPT
    payload = {
        "campaign_summary": campaign_summary.to_dict("records"),
        "budget": budget_info,
        "keyword_rollup": keyword_rollup,
        "best_keywords": best_keywords,
        "worst_keywords": worst_keywords,
    }

    openai_client = openai.OpenAI(api_key=api_key)
    prompt = build_prompt(payload)
    
    # Non-streaming call for backward compatibility
    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a concise Amazon Ads optimization assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    recommendations_text = response.choices[0].message.content

    # Save recommendations to markdown file
    output_file = save_recommendations_to_file(campaign, recommendations_text)
    
    return recommendations_text, str(output_file)


def save_recommendations_to_file(campaign: str, recommendations_text: str) -> Path:
    """Save recommendations to a markdown file with proper formatting."""
    output_dir = Path("./output/recommendations")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_campaign = "".join(c if c.isalnum() or c in "-_ " else "_" for c in campaign)[:50]
    output_file = output_dir / f"recommendations_{safe_campaign}_{timestamp}.md"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# Campaign Recommendations: {campaign}\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        f.write(recommendations_text)
    
    return output_file


def generate_recommendations_streaming(
    campaign: str,
    ts_host: str,
    ts_port: int,
    ts_protocol: str,
    ts_api_key: str,
    api_key: str,
    model: str = "gpt-4o-mini",
    top_n: int = 10,
    min_impr: int = 50,
):
    """
    Generator that yields recommendation text chunks as they stream from GPT.
    
    Usage:
        full_text = ""
        for chunk in generate_recommendations_streaming(...):
            full_text += chunk
            print(chunk, end="", flush=True)
    
    After iteration completes, returns (full_text, output_file_path) via final yield.
    """
    if not api_key:
        raise ValueError("Missing OpenAI API key.")
    if not ts_api_key:
        raise ValueError("Missing Typesense API key.")

    client = typesense.Client(
        {
            "nodes": [{"host": ts_host, "port": ts_port, "protocol": ts_protocol}],
            "api_key": ts_api_key,
            "connection_timeout_seconds": 5,
        }
    )

    collections_map = {
        "search_terms": find_collection(client, "search_term"),
        "campaigns": find_collection(client, "campaign"),
        "targeting": find_collection(client, "targeting"),
        "placement": find_collection(client, "placement"),
        "budgets": find_collection(client, "budget"),
    }

    reports = {k: fetch_collection_df(client, v) for k, v in collections_map.items()}

    # Campaign filter
    campaign_name = campaign.lower()
    search_terms = reports["search_terms"]
    campaigns_df = reports["campaigns"]
    budgets = reports.get("budgets", pd.DataFrame())

    st_camp_col = find_campaign_column(search_terms)
    camp_camp_col = find_campaign_column(campaigns_df)

    search_terms = search_terms[search_terms[st_camp_col].str.lower() == campaign_name]
    campaigns_df = campaigns_df[campaigns_df[camp_camp_col].str.lower() == campaign_name]

    if budgets is not None and not budgets.empty:
        try:
            budget_camp_col = find_campaign_column(budgets)
            budgets = budgets[budgets[budget_camp_col].str.lower() == campaign_name]
        except KeyError:
            budgets = pd.DataFrame()

    if campaigns_df.empty:
        raise ValueError(f"Campaign not found: {campaign}")

    campaigns_df = normalize_metrics(campaigns_df)

    campaign_summary = campaigns_df.groupby(camp_camp_col, as_index=False).agg(
        spend=("spend", "sum"),
        sales=("sales", "sum"),
        clicks=("clicks", "sum"),
        impressions=("impressions", "sum"),
    )
    if "budget_amount" in campaigns_df.columns:
        campaign_summary["budget_amount"] = campaigns_df["budget_amount"].fillna(0).iloc[0] if len(campaigns_df) > 0 else 0
    campaign_summary["ctr"] = (campaign_summary["clicks"] / campaign_summary["impressions"].replace(0, 1)) * 100
    campaign_summary["cvr"] = campaign_summary["sales"] / campaign_summary["clicks"].replace(0, 1)
    campaign_summary["acos"] = (campaign_summary["spend"] / campaign_summary["sales"].replace(0, 1)) * 100
    campaign_summary["roas"] = campaign_summary["sales"] / campaign_summary["spend"].replace(0, 1)
    campaign_summary = campaign_summary.replace([float("inf"), float("-inf")], 0)

    budget_info = {}
    if budgets is not None and not budgets.empty:
        budgets = normalize_metrics(budgets)
        budget_amt = float(budgets["budget_amount"].sum()) if "budget_amount" in budgets.columns else 0
        budget_info = {
            "total_budget": budget_amt,
            "daily_budget": budget_amt / max(len(budgets), 1) if budget_amt > 0 else 0,
            "rows": budgets.head(5).to_dict("records"),
        }
    else:
        budget_amt = float(campaign_summary["budget_amount"].iloc[0]) if "budget_amount" in campaign_summary.columns else 0
        budget_info = {"total_budget": budget_amt, "daily_budget": budget_amt}

    keyword_rollup, best_keywords, worst_keywords = compute_keyword_metrics(
        search_terms, min_impr=min_impr, top_n=top_n
    )

    payload = {
        "campaign_summary": campaign_summary.to_dict("records"),
        "budget": budget_info,
        "keyword_rollup": keyword_rollup,
        "best_keywords": best_keywords,
        "worst_keywords": worst_keywords,
    }

    openai_client = openai.OpenAI(api_key=api_key)
    prompt = build_prompt(payload)
    
    # Streaming call
    stream = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a concise Amazon Ads optimization assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        stream=True,
    )
    
    full_text = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            full_text += content
            yield content
    
    # Save to file after streaming completes
    output_file = save_recommendations_to_file(campaign, full_text)
    
    # Final yield with metadata (caller can check for dict type)
    yield {"complete": True, "output_file": str(output_file), "full_text": full_text}


if __name__ == "__main__":
    main()
