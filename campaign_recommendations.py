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
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import openai
import pandas as pd
import typesense
from dotenv import load_dotenv
from token_tracker import get_tracker, extract_token_usage, estimate_tokens

# Load environment variables from .env file
load_dotenv()

# ==================== LOGGING SETUP ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("recommendations")


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


def compute_keyword_metrics(
    search_terms_df: pd.DataFrame,
    targeting_df: pd.DataFrame = None,
    min_impr: int = 50,
    top_n: int = 10,
):
    """
    Analyze both customer search terms and targeting keywords.
    
    Args:
        search_terms_df: DataFrame with customer search terms (what customers searched)
        targeting_df: DataFrame with targeting keywords (what we're bidding on)
        min_impr: Minimum impressions threshold
        top_n: Number of top/bottom keywords to return
    
    Returns:
        Tuple of (keyword_rollup, best_keywords, worst_keywords, targeting_analysis, alignment_analysis)
    """
    logger.info("=" * 60)
    logger.info("COMPUTING KEYWORD METRICS")
    logger.info("=" * 60)
    
    # Normalize both dataframes
    search_terms_df = normalize_metrics(search_terms_df.copy())
    if targeting_df is not None and not targeting_df.empty:
        targeting_df = normalize_metrics(targeting_df.copy())
    
    # ==================== CUSTOMER SEARCH TERMS ANALYSIS ====================
    logger.info("Analyzing customer search terms (what customers actually searched)...")
    search_term_col = find_keyword_column(search_terms_df)
    logger.info(f"Customer search term column: {search_term_col}")
    
    customer_search_agg = (
        search_terms_df.groupby(search_term_col, as_index=False)
        .agg(
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
            spend=("spend", "sum"),
            sales=("sales", "sum"),
        )
    )
    
    customer_search_agg["ctr"] = (customer_search_agg["clicks"] / customer_search_agg["impressions"].replace(0, 1)) * 100
    customer_search_agg["cpc"] = customer_search_agg["spend"] / customer_search_agg["clicks"].replace(0, 1)
    customer_search_agg["acos"] = (customer_search_agg["spend"] / customer_search_agg["sales"].replace(0, 1)) * 100
    customer_search_agg["roas"] = customer_search_agg["sales"] / customer_search_agg["spend"].replace(0, 1)
    customer_search_agg = customer_search_agg.replace([float("inf"), float("-inf")], 0)
    customer_search_agg["type"] = "customer_search"
    
    customer_filtered = customer_search_agg[customer_search_agg["impressions"] >= min_impr].copy()
    logger.info(f"  - Total customer search terms: {len(customer_search_agg)}")
    logger.info(f"  - Customer search terms with {min_impr}+ impressions: {len(customer_filtered)}")
    
    # ==================== TARGETING KEYWORDS ANALYSIS ====================
    targeting_agg = None
    targeting_filtered = None
    
    if targeting_df is not None and not targeting_df.empty:
        logger.info("Analyzing targeting keywords (what we're bidding on)...")
        targeting_col = find_keyword_column(targeting_df)
        logger.info(f"Targeting keyword column: {targeting_col}")
        
        targeting_agg = (
            targeting_df.groupby(targeting_col, as_index=False)
            .agg(
                impressions=("impressions", "sum"),
                clicks=("clicks", "sum"),
                spend=("spend", "sum"),
                sales=("sales", "sum"),
            )
        )
        
        targeting_agg["ctr"] = (targeting_agg["clicks"] / targeting_agg["impressions"].replace(0, 1)) * 100
        targeting_agg["cpc"] = targeting_agg["spend"] / targeting_agg["clicks"].replace(0, 1)
        targeting_agg["acos"] = (targeting_agg["spend"] / targeting_agg["sales"].replace(0, 1)) * 100
        targeting_agg["roas"] = targeting_agg["sales"] / targeting_agg["spend"].replace(0, 1)
        targeting_agg = targeting_agg.replace([float("inf"), float("-inf")], 0)
        targeting_agg["type"] = "targeting"
        
        targeting_filtered = targeting_agg[targeting_agg["impressions"] >= min_impr].copy()
        logger.info(f"  - Total targeting keywords: {len(targeting_agg)}")
        logger.info(f"  - Targeting keywords with {min_impr}+ impressions: {len(targeting_filtered)}")
    else:
        logger.warning("  - No targeting data available")
    
    # ==================== ALIGNMENT ANALYSIS ====================
    alignment_analysis = {
        "well_aligned": [],
        "poorly_aligned": [],
        "untargeted_searches": [],
        "unused_targeting": [],
    }
    
    if targeting_agg is not None and not targeting_agg.empty:
        logger.info("Analyzing alignment between targeting and customer searches...")
        
        # Normalize keyword names for comparison (lowercase, strip whitespace)
        customer_terms = set(customer_search_agg[search_term_col].str.lower().str.strip())
        targeting_terms = set(targeting_agg[targeting_col].str.lower().str.strip())
        
        # Find well-aligned: targeting keywords that match customer searches
        well_aligned_terms = customer_terms.intersection(targeting_terms)
        logger.info(f"  - Well-aligned keywords (targeted + searched): {len(well_aligned_terms)}")
        
        # Find untargeted searches: popular customer searches not being targeted
        untargeted_searches = customer_terms - targeting_terms
        untargeted_df = customer_search_agg[
            customer_search_agg[search_term_col].str.lower().str.strip().isin(untargeted_searches)
        ].copy()
        untargeted_df = untargeted_df[untargeted_df["impressions"] >= min_impr].copy()
        untargeted_df = untargeted_df.sort_values(["impressions", "sales"], ascending=[False, False]).head(top_n)
        alignment_analysis["untargeted_searches"] = untargeted_df.to_dict("records")
        logger.info(f"  - Untargeted customer searches (opportunities): {len(untargeted_df)}")
        
        # Find unused targeting: targeting keywords that don't match customer searches
        unused_targeting = targeting_terms - customer_terms
        unused_df = targeting_agg[
            targeting_agg[targeting_col].str.lower().str.strip().isin(unused_targeting)
        ].copy()
        unused_df = unused_df[unused_df["impressions"] >= min_impr].copy()
        unused_df = unused_df.sort_values(["spend", "acos"], ascending=[False, True]).head(top_n)
        alignment_analysis["unused_targeting"] = unused_df.to_dict("records")
        logger.info(f"  - Unused targeting keywords (not matching searches): {len(unused_df)}")
        
        # Well-aligned: targeting keywords that match customer searches and perform well
        well_aligned_df = targeting_agg[
            targeting_agg[targeting_col].str.lower().str.strip().isin(well_aligned_terms)
        ].copy()
        well_aligned_df = well_aligned_df[well_aligned_df["impressions"] >= min_impr].copy()
        well_aligned_df = well_aligned_df.sort_values(["roas", "sales"], ascending=[False, False]).head(top_n)
        alignment_analysis["well_aligned"] = well_aligned_df.to_dict("records")
        
        # Poorly-aligned: targeting keywords that match searches but perform poorly (high ACOS, low ROAS)
        poorly_aligned_df = targeting_agg[
            targeting_agg[targeting_col].str.lower().str.strip().isin(well_aligned_terms)
        ].copy()
        poorly_aligned_df = poorly_aligned_df[poorly_aligned_df["impressions"] >= min_impr].copy()
        # Filter for poor performance: ACOS > 30% or ROAS < 1.0
        poorly_aligned_df = poorly_aligned_df[
            (poorly_aligned_df["acos"] > 30) | (poorly_aligned_df["roas"] < 1.0)
        ].copy()
        poorly_aligned_df = poorly_aligned_df.sort_values(["acos", "spend"], ascending=[False, False]).head(top_n)
        alignment_analysis["poorly_aligned"] = poorly_aligned_df.to_dict("records")
        logger.info(f"  - Poorly-aligned targeting (high ACOS/low ROAS): {len(poorly_aligned_df)}")
    
    # ==================== COMBINED KEYWORD ROLLUP ====================
    # Combine both types for overall rollup
    keyword_rollup = customer_search_agg.copy()
    if targeting_agg is not None and not targeting_agg.empty:
        # Rename columns to match
        targeting_rollup = targeting_agg.copy()
        targeting_rollup = targeting_rollup.rename(columns={targeting_col: search_term_col})
        keyword_rollup = pd.concat([keyword_rollup, targeting_rollup], ignore_index=True)
        # Aggregate duplicates (if same keyword appears in both)
        keyword_rollup = keyword_rollup.groupby(search_term_col, as_index=False).agg({
            "impressions": "sum",
            "clicks": "sum",
            "spend": "sum",
            "sales": "sum",
            "ctr": "mean",  # Average CTR
            "cpc": "mean",
            "acos": "mean",
            "roas": "mean",
        })
    
    # ==================== BEST/WORST KEYWORDS ====================
    filtered = keyword_rollup[keyword_rollup["impressions"] >= min_impr].copy()
    
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
    
    logger.info("=" * 60)
    logger.info("KEYWORD METRICS COMPLETE")
    logger.info("=" * 60)
    
    return (
        keyword_rollup.to_dict("records"),
        best,
        worst,
        {
            "targeting_keywords": targeting_agg.to_dict("records") if targeting_agg is not None else [],
            "customer_search_terms": customer_search_agg.to_dict("records"),
        },
        alignment_analysis,
    )


def build_prompt(data: Dict[str, Any]) -> str:
    """Build a detailed, quantitative prompt for GPT with precomputed metrics."""
    # return (
    #     "You are an Amazon Sponsored Products optimization expert with deep expertise in PPC analytics.\n\n"
    
    #     "# YOUR TASK\n"
    #     "Analyze the campaign data below and provide 8-12 specific, quantitative recommendations.\n"
    #     "Each recommendation must include:\n"
    #     "- Exact numbers (current vs. target metrics)\n"
    #     "- Expected impact (e.g., 'reduce wasted spend by $X' or 'increase CTR from X% to Y%')\n"
    #     "- Specific action steps with thresholds\n"
    #     "- Priority level (High/Medium/Low) based on potential ROI impact\n\n"
      
    #     "# CAMPAIGN DATA\n"
    #     f"Campaign Summary:\n{json.dumps(data['campaign_summary'], indent=2, ensure_ascii=False)}\n\n"
       
    #     f"Budget Metrics:\n{json.dumps(data['budget'], indent=2, ensure_ascii=False)}\n\n"
    
    #     f"Top Performing Keywords (High CTR):\n{json.dumps(data['best_keywords'], indent=2, ensure_ascii=False)}\n\n"
      
    #     f"Underperforming Keywords (Low CTR):\n{json.dumps(data['worst_keywords'], indent=2, ensure_ascii=False)}\n\n"
        
    #     f"Complete Keyword Performance Rollup:\n{json.dumps(data['keyword_rollup'], indent=2, ensure_ascii=False)[:5000]}\n\n"
    # )
    
    # Add keyword type analysis if available
    prompt_additions = ""
    if 'keyword_types' in data and data['keyword_types']:
        prompt_additions += (
            f"\n# UNDERSTANDING KEYWORD TYPES - CRITICAL CONCEPT\n"
            f"Amazon Ads has TWO distinct types of keywords that you MUST understand:\n\n"
            f"1. **TARGETING KEYWORDS** (Set in Campaign Manager):\n"
            f"   - These are keywords YOU actively bid on and set in your campaign manager\n"
            f"   - They represent your targeting strategy - what you WANT to show ads for\n"
            f"   - Example: If you set 'wireless headphones' as a targeting keyword, you're telling Amazon to show your ad when that keyword is relevant\n"
            f"   - Performance metrics show how well your chosen targeting keywords are performing\n\n"
            f"2. **CUSTOMER SEARCH TERMS** (What Customers Actually Searched):\n"
            f"   - These are the actual search queries customers typed into Amazon\n"
            f"   - They represent REAL customer behavior - what customers are ACTUALLY searching for\n"
            f"   - Example: A customer might search 'best bluetooth earbuds' and your ad appeared (even if you targeted 'wireless headphones')\n"
            f"   - This shows the gap between what you're targeting vs. what customers want\n\n"
            f"**WHY THIS MATTERS:**\n"
            f"- If customers search for terms you're NOT targeting, you're missing opportunities\n"
            f"- If you're targeting keywords customers DON'T search for, you're wasting budget\n"
            f"- The goal is to align your targeting keywords with actual customer search behavior\n\n"
            f"Targeting Keywords Data (what we're bidding on):\n"
            f"{json.dumps(data['keyword_types'].get('targeting_keywords', [])[:20], indent=2, ensure_ascii=False)}\n\n"
            f"Customer Search Terms Data (what customers actually searched):\n"
            f"{json.dumps(data['keyword_types'].get('customer_search_terms', [])[:20], indent=2, ensure_ascii=False)}\n\n"
        )
    
    # Add alignment analysis if available
    if 'alignment_analysis' in data and data['alignment_analysis']:
        alignment = data['alignment_analysis']
        prompt_additions += (
            f"\n# TARGETING ALIGNMENT ANALYSIS\n"
            f"This analysis compares your TARGETING KEYWORDS (what you set in campaign manager) vs CUSTOMER SEARCH TERMS (what customers actually searched).\n"
            f"This reveals whether your targeting strategy aligns with real customer behavior.\n\n"
        )
        
        if alignment.get('well_aligned'):
            prompt_additions += (
                f"✅ Well-Aligned Targeting Keywords (matching customer searches, performing well):\n"
                f"{json.dumps(alignment['well_aligned'], indent=2, ensure_ascii=False)}\n\n"
            )
        
        if alignment.get('poorly_aligned'):
            prompt_additions += (
                f"⚠️ Poorly-Aligned Targeting Keywords (matching searches but high ACOS/low ROAS):\n"
                f"{json.dumps(alignment['poorly_aligned'], indent=2, ensure_ascii=False)}\n\n"
            )
        
        if alignment.get('untargeted_searches'):
            prompt_additions += (
                f"🎯 Untargeted Customer Searches (popular searches we're NOT bidding on - OPPORTUNITIES):\n"
                f"{json.dumps(alignment['untargeted_searches'], indent=2, ensure_ascii=False)}\n\n"
                f"**These are high-priority keywords to add to targeting - customers are searching for them but we're missing out.**\n\n"
            )
        
        if alignment.get('unused_targeting'):
            prompt_additions += (
                f"❌ Unused Targeting Keywords (we're bidding on these but customers aren't searching for them):\n"
                f"{json.dumps(alignment['unused_targeting'], indent=2, ensure_ascii=False)}\n\n"
                f"**Consider pausing or reducing bids on these - they don't align with customer search behavior.**\n\n"
            )
    
    return (
        "You are an Amazon Sponsored Products optimization expert with deep expertise in PPC analytics.\n\n"
        
        "# CRITICAL CONCEPT: TWO TYPES OF KEYWORDS\n"
        "Amazon Ads tracks TWO distinct keyword types that you MUST understand:\n\n"
        "1. **TARGETING KEYWORDS** (Set in Campaign Manager):\n"
        "   - Keywords YOU actively set and bid on in your campaign manager\n"
        "   - Represent your targeting STRATEGY - what you want to show ads for\n"
        "   - These are under YOUR CONTROL - you decide which ones to add, pause, or adjust bids\n"
        "   - Example: You set 'wireless headphones' as a targeting keyword with a $1.50 bid\n\n"
        "2. **CUSTOMER SEARCH TERMS** (What Customers Actually Searched):\n"
        "   - The actual search queries customers typed into Amazon's search box\n"
        "   - Represent REAL customer BEHAVIOR - what customers are actually looking for\n"
        "   - These are NOT under your control - they show what customers want\n"
        "   - Example: A customer searches 'best bluetooth earbuds' and your ad appeared\n"
        "   - Your ad appeared because Amazon matched it to your targeting keywords (e.g., 'wireless headphones')\n\n"
        "**THE KEY INSIGHT:**\n"
        "- If customers search for terms you're NOT targeting → You're missing opportunities (add these as targeting keywords)\n"
        "- If you're targeting keywords customers DON'T search for → You're wasting budget (pause or reduce bids)\n"
        "- The goal: Align your TARGETING KEYWORDS with actual CUSTOMER SEARCH TERMS\n\n"
        
        "# YOUR TASK\n"
        "Analyze the campaign data below and provide 8-12 specific, quantitative recommendations.\n"
        "Each recommendation must include:\n"
        "- Exact numbers (current vs. target metrics)\n"
        "- Expected impact (e.g., 'reduce wasted spend by $X' or 'increase CTR from X% to Y%')\n"
        "- Specific action steps with thresholds\n"
        "- Priority level (High/Medium/Low) based on potential ROI impact\n"
        "- **SPECIAL FOCUS**: Use the alignment analysis to recommend adding untargeted customer searches and pausing unused targeting keywords\n\n"
        
        "# CAMPAIGN DATA\n"
        "**CRITICAL: METRIC INTERPRETATION**\n"
        "- Metrics in 'Campaign Summary' (spend, sales, clicks, impressions) are **DAILY AVERAGES**, not totals\n"
        "- These are calculated by dividing total metrics by the number of days in the date range\n"
        "- The date range (date_range_start, date_range_end) and number of days (num_days) are included in the summary\n"
        "- Total values over the entire period are also provided (spend_total, sales_total, clicks_total, impressions_total)\n"
        "- **Always use daily averages** when making recommendations (e.g., 'daily spend of $X', 'daily impressions of Y')\n"
        "- When projecting monthly impact, multiply daily averages by 30, not by the actual number of days in the report\n\n"
        f"Campaign Summary:\n{json.dumps(data['campaign_summary'], indent=2, ensure_ascii=False)}\n\n"
        
        f"Budget Metrics:\n{json.dumps(data['budget'], indent=2, ensure_ascii=False)}\n\n"
        
        f"Top Performing Keywords (High CTR):\n{json.dumps(data['best_keywords'], indent=2, ensure_ascii=False)}\n\n"
        
        f"Underperforming Keywords (Low CTR):\n{json.dumps(data['worst_keywords'], indent=2, ensure_ascii=False)}\n\n"
        
        f"Complete Keyword Performance Rollup:\n{json.dumps(data['keyword_rollup'], indent=2, ensure_ascii=False)[:5000]}\n\n"
        
        f"{prompt_additions}"
        
        "# ANALYSIS FRAMEWORK\n"
        "Structure your recommendations across these categories:\n\n"
        
        "## 1. BUDGET OPTIMIZATION\n"
        "- Calculate budget utilization rate and daily pacing\n"
        "- Identify if campaign is budget-constrained (>90% spend) or under-utilizing (<70% spend)\n"
        "- Recommend specific budget adjustments with dollar amounts and rationale\n"
        "- Project impact on impression share and potential sales\n\n"
        "- Do no exceed new budget more than 30% of the current budget"
        
        "## 2. BID STRATEGY\n"
        "- Analyze current avg CPC vs. category benchmarks\n"
        "- Calculate bid efficiency: (Conversions × AOV) / Cost\n"
        "- Recommend specific bid increases/decreases by keyword or ad group (e.g., 'increase bid by $0.15 for keyword X')\n"
        "- Identify keywords with high CVR but low impression share (bid too low)\n\n"
        "- Do no exceed new bid more than 30% of the current bid"
        "- Do not overestimate the impact of bid changes on sales impressions and conversions"
        
        "## 3. PLACEMENT OPTIMIZATION\n"
        "- Compare performance across Top of Search, Product Pages, and Rest of Search\n"
        "- Calculate placement-specific ACOS and conversion rates\n"
        "- Recommend bid multiplier adjustments (e.g., 'increase Top of Search multiplier from X% to Y%')\n"
        "- Quantify expected cost and revenue impact\n\n"
        
        "## 4. KEYWORD EXPANSION & TARGETING ALIGNMENT\n"
        "**UNDERSTAND THE TWO KEYWORD TYPES:**\n"
        "- **Targeting Keywords**: Keywords YOU set in campaign manager (your strategy)\n"
        "- **Customer Search Terms**: What customers ACTUALLY searched (real behavior)\n\n"
        "**KEY INSIGHTS TO PROVIDE:**\n"
        "- **CRITICAL PRIORITY**: 'Untargeted Customer Searches' - customers are searching for these but you're NOT targeting them\n"
        "  → These are HIGH-VALUE OPPORTUNITIES - add them as exact-match targeting keywords immediately\n"
        "  → Calculate potential revenue: If you add keyword X (with Y impressions, Z% CTR, $W sales), project incremental sales\n"
        "- **COST SAVINGS**: 'Unused Targeting Keywords' - you're bidding on these but customers DON'T search for them\n"
        "  → These are WASTING BUDGET - recommend pausing or reducing bids by X%\n"
        "  → Calculate wasted spend: If keyword X has $Y spend but 0 customer searches, that's $Y wasted per period\n"
        "- **PERFORMANCE OPTIMIZATION**: 'Well-Aligned' vs 'Poorly-Aligned' targeting keywords\n"
        "  → Well-aligned: Keep and potentially increase bids (they match customer behavior AND perform well)\n"
        "  → Poorly-aligned: These match customer searches but have high ACOS - recommend bid reductions or negative keywords\n"
        "- **ACTIONABLE RECOMMENDATIONS:**\n"
        "  → List specific untargeted customer searches to ADD (with suggested bids, match types, ad groups)\n"
        "  → List specific unused targeting keywords to PAUSE (with estimated monthly savings)\n"
        "  → Recommend bid adjustments for poorly-aligned keywords (reduce by $X or Y%)\n"
        "  → Project impact: 'Adding these 5 untargeted searches could capture $X additional monthly sales'\n\n"
        
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
    parser.add_argument("--model", default="gpt-5-mini", help="OpenAI chat model.")
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
    logger.debug(f"Looking for collection containing '{pattern}'")
    cols = client.collections.retrieve()
    for c in cols:
        if pattern in c["name"]:
            logger.debug(f"Found collection: {c['name']}")
            return c["name"]
    raise FileNotFoundError(f"Missing collection containing '{pattern}' in Typesense")


def fetch_collection_df(client: typesense.Client, name: str) -> pd.DataFrame:
    logger.info(f"Exporting collection '{name}'...")
    export_str = client.collections[name].documents.export()
    if not export_str:
        logger.warning(f"Collection '{name}' is empty")
        return pd.DataFrame()
    lines = [json.loads(line) for line in export_str.splitlines() if line.strip()]
    logger.info(f"Exported {len(lines)} rows from '{name}'")
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


def find_date_column(df: pd.DataFrame) -> Optional[str]:
    """Find the date column name in the dataframe."""
    # Common date column names
    candidates = [
        "date", "start_date", "end_date", "reporting_date", "date_range_start",
        "date_range_end", "start_time", "end_time", "day", "report_date"
    ]
    
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    
    # Fuzzy match: look for columns with "date" in the name
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            return col
    
    return None


def calculate_date_range(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], int]:
    """
    Calculate the date range from a dataframe.
    
    Returns:
        Tuple of (start_date, end_date, num_days)
        Returns (None, None, 1) if no date column found
    """
    date_col = find_date_column(df)
    
    if date_col is None or date_col not in df.columns:
        logger.warning("No date column found in dataframe, assuming 1 day period")
        return None, None, 1
    
    try:
        # Try to parse dates
        dates = pd.to_datetime(df[date_col], errors='coerce')
        dates = dates.dropna()
        
        if len(dates) == 0:
            logger.warning(f"Date column '{date_col}' found but no valid dates, assuming 1 day period")
            return None, None, 1
        
        start_date = dates.min()
        end_date = dates.max()
        
        # Calculate number of days (inclusive)
        num_days = (end_date - start_date).days + 1
        
        if num_days < 1:
            num_days = 1
        
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        logger.info(f"Date range detected: {start_str} to {end_str} ({num_days} days)")
        
        return start_str, end_str, num_days
        
    except Exception as e:
        logger.warning(f"Error parsing dates from column '{date_col}': {e}, assuming 1 day period")
        return None, None, 1


def generate_recommendations(
    campaign: str,
    ts_host: str,
    ts_port: int,
    ts_protocol: str,
    ts_api_key: str,
    api_key: str,
    model: str = "gpt-5-mini",
    top_n: int = 10,
    min_impr: int = 50,
) -> str:
    """Compute metrics from Typesense collections and return GPT recommendations as text."""
    logger.info("=" * 60)
    logger.info(f"GENERATING RECOMMENDATIONS FOR: {campaign}")
    logger.info("=" * 60)
    
    if not api_key:
        raise ValueError("Missing OpenAI API key.")
    if not ts_api_key:
        raise ValueError("Missing Typesense API key.")

    logger.info(f"Connecting to Typesense at {ts_protocol}://{ts_host}:{ts_port}")
    client = typesense.Client(
        {
            "nodes": [{"host": ts_host, "port": ts_port, "protocol": ts_protocol}],
            "api_key": ts_api_key,
            "connection_timeout_seconds": 5,
        }
    )

    logger.info("Step 1: Finding required collections...")
    collections_map = {
        "search_terms": find_collection(client, "search_term"),
        "campaigns": find_collection(client, "campaign"),
        "targeting": find_collection(client, "targeting"),
        "placement": find_collection(client, "placement"),
        "budgets": find_collection(client, "budget"),
    }
    logger.info(f"Collections found: {list(collections_map.values())}")

    logger.info("Step 2: Fetching data from collections...")
    reports = {k: fetch_collection_df(client, v) for k, v in collections_map.items()}

    # Campaign filter
    campaign_name = campaign.lower()
    search_terms = reports["search_terms"]
    campaigns_df = reports["campaigns"]
    budgets = reports.get("budgets", pd.DataFrame())

    logger.info(f"Step 3: Filtering data for campaign '{campaign}'...")
    st_camp_col = find_campaign_column(search_terms)
    camp_camp_col = find_campaign_column(campaigns_df)

    search_terms = search_terms[search_terms[st_camp_col].str.lower() == campaign_name]
    campaigns_df = campaigns_df[campaigns_df[camp_camp_col].str.lower() == campaign_name]
    logger.info(f"  - Search terms rows: {len(search_terms)}")
    logger.info(f"  - Campaign rows: {len(campaigns_df)}")

    if budgets is not None and not budgets.empty:
        try:
            budget_camp_col = find_campaign_column(budgets)
            budgets = budgets[budgets[budget_camp_col].str.lower() == campaign_name]
            logger.info(f"  - Budget rows: {len(budgets)}")
        except KeyError:
            logger.warning("  - No budget column found")
            budgets = pd.DataFrame()

    if campaigns_df.empty:
        logger.error(f"Campaign not found: {campaign}")
        raise ValueError(f"Campaign not found: {campaign}")

    # Normalize metric columns (handles 7_day_total_sales -> sales, etc.)
    campaigns_df = normalize_metrics(campaigns_df)

    logger.info("Step 4: Calculating date range...")
    # Try to find date range from campaigns_df, fallback to search_terms
    start_date, end_date, num_days = calculate_date_range(campaigns_df)
    if start_date is None:
        start_date, end_date, num_days = calculate_date_range(search_terms)
    
    logger.info(f"  - Date range: {start_date or 'Unknown'} to {end_date or 'Unknown'} ({num_days} days)")

    logger.info("Step 5: Computing campaign metrics...")
    campaign_summary = campaigns_df.groupby(camp_camp_col, as_index=False).agg(
        spend=("spend", "sum"),
        sales=("sales", "sum"),
        clicks=("clicks", "sum"),
        impressions=("impressions", "sum"),
    )
    if "budget_amount" in campaigns_df.columns:
        campaign_summary["budget_amount"] = campaigns_df["budget_amount"].fillna(0).iloc[0] if len(campaigns_df) > 0 else 0
    
    # Convert totals to daily averages
    campaign_summary["spend_total"] = campaign_summary["spend"]
    campaign_summary["sales_total"] = campaign_summary["sales"]
    campaign_summary["clicks_total"] = campaign_summary["clicks"]
    campaign_summary["impressions_total"] = campaign_summary["impressions"]
    
    campaign_summary["spend"] = campaign_summary["spend"] / num_days
    campaign_summary["sales"] = campaign_summary["sales"] / num_days
    campaign_summary["clicks"] = campaign_summary["clicks"] / num_days
    campaign_summary["impressions"] = campaign_summary["impressions"] / num_days
    
    # Add date range info
    campaign_summary["date_range_start"] = start_date or "Unknown"
    campaign_summary["date_range_end"] = end_date or "Unknown"
    campaign_summary["num_days"] = num_days
    
    campaign_summary["ctr"] = (campaign_summary["clicks"] / campaign_summary["impressions"].replace(0, 1)) * 100
    campaign_summary["cvr"] = campaign_summary["sales"] / campaign_summary["clicks"].replace(0, 1)
    campaign_summary["acos"] = (campaign_summary["spend"] / campaign_summary["sales"].replace(0, 1)) * 100
    campaign_summary["roas"] = campaign_summary["sales"] / campaign_summary["spend"].replace(0, 1)
    campaign_summary = campaign_summary.replace([float("inf"), float("-inf")], 0)
    
    summary = campaign_summary.iloc[0] if len(campaign_summary) > 0 else {}
    logger.info(f"  - Total Spend: ${summary.get('spend_total', 0):.2f} over {num_days} days")
    logger.info(f"  - Daily Spend: ${summary.get('spend', 0):.2f}")
    logger.info(f"  - Total Sales: ${summary.get('sales_total', 0):.2f} over {num_days} days")
    logger.info(f"  - Daily Sales: ${summary.get('sales', 0):.2f}")
    logger.info(f"  - Total Impressions: {summary.get('impressions_total', 0):,.0f} over {num_days} days")
    logger.info(f"  - Daily Impressions: {summary.get('impressions', 0):,.0f}")
    logger.info(f"  - ACOS: {summary.get('acos', 0):.2f}%")
    logger.info(f"  - ROAS: {summary.get('roas', 0):.2f}x")

    logger.info("Step 6: Processing budget data...")
    budget_info = {}
    if budgets is not None and not budgets.empty:
        budgets = normalize_metrics(budgets)
        logger.debug(f"Budget columns: {list(budgets.columns)}")
        
        budget_amt = 0
        if "budget_amount" in budgets.columns:
            budget_amt = float(budgets["budget_amount"].sum())
        
        budget_info = {
            "total_budget": budget_amt,
            "daily_budget": budget_amt / max(len(budgets), 1) if budget_amt > 0 else 0,
            "rows": budgets.head(5).to_dict("records"),
        }
        logger.info(f"  - Total budget: ${budget_amt:.2f}")
    else:
        budget_amt = float(campaign_summary["budget_amount"].iloc[0]) if "budget_amount" in campaign_summary.columns else 0
        budget_info = {"total_budget": budget_amt, "daily_budget": budget_amt}
        logger.info(f"  - Budget from campaign: ${budget_amt:.2f}")

    logger.info("Step 7: Analyzing keywords...")
    targeting_data = reports.get("targeting", pd.DataFrame())
    if targeting_data is not None and not targeting_data.empty:
        targeting_camp_col = find_campaign_column(targeting_data)
        targeting_data = targeting_data[targeting_data[targeting_camp_col].str.lower() == campaign_name]
        logger.info(f"  - Targeting rows: {len(targeting_data)}")
    else:
        targeting_data = pd.DataFrame()
    
    keyword_rollup, best_keywords, worst_keywords, keyword_types, alignment_analysis = compute_keyword_metrics(
        search_terms_df=search_terms,
        targeting_df=targeting_data,
        min_impr=min_impr,
        top_n=top_n,
    )
    logger.info(f"  - Total keywords: {len(keyword_rollup)}")
    logger.info(f"  - Best keywords: {len(best_keywords)}")
    logger.info(f"  - Worst keywords: {len(worst_keywords)}")
    logger.info(f"  - Well-aligned targeting: {len(alignment_analysis.get('well_aligned', []))}")
    logger.info(f"  - Untargeted customer searches: {len(alignment_analysis.get('untargeted_searches', []))}")

    logger.info("Step 8: Building GPT prompt...")
    payload = {
        "campaign_summary": campaign_summary.to_dict("records"),
        "budget": budget_info,
        "keyword_rollup": keyword_rollup,
        "best_keywords": best_keywords,
        "worst_keywords": worst_keywords,
        "keyword_types": keyword_types,
        "alignment_analysis": alignment_analysis,
    }
    prompt = build_prompt(payload)
    logger.info(f"  - Prompt length: {len(prompt)} chars")

    logger.info(f"Step 9: Calling OpenAI API (model: {model})...")
    openai_client = openai.OpenAI(api_key=api_key)
    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a concise Amazon Ads optimization assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    recommendations_text = response.choices[0].message.content
    logger.info(f"  - Response received: {len(recommendations_text)} chars")
    
    # Track token usage
    token_usage = extract_token_usage(response)
    if token_usage:
        tracker = get_tracker()
        tracker.record_usage(
            input_tokens=token_usage["input_tokens"],
            output_tokens=token_usage["output_tokens"],
            total_tokens=token_usage["total_tokens"],
            model=model,
            campaign=campaign,
            metadata={"function": "generate_recommendations", "streaming": False},
        )

    logger.info("Step 10: Saving recommendations...")
    output_file = save_recommendations_to_file(campaign, recommendations_text)
    logger.info(f"  - Saved to: {output_file}")
    logger.info("=" * 60)
    logger.info("RECOMMENDATIONS COMPLETE")
    logger.info("=" * 60)
    
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
    """
    logger.info("=" * 60)
    logger.info(f"STREAMING RECOMMENDATIONS FOR: {campaign}")
    logger.info("=" * 60)
    
    if not api_key:
        raise ValueError("Missing OpenAI API key.")
    if not ts_api_key:
        raise ValueError("Missing Typesense API key.")

    logger.info(f"Connecting to Typesense at {ts_protocol}://{ts_host}:{ts_port}")
    client = typesense.Client(
        {
            "nodes": [{"host": ts_host, "port": ts_port, "protocol": ts_protocol}],
            "api_key": ts_api_key,
            "connection_timeout_seconds": 5,
        }
    )

    logger.info("Finding collections...")
    collections_map = {
        "search_terms": find_collection(client, "search_term"),
        "campaigns": find_collection(client, "campaign"),
        "targeting": find_collection(client, "targeting"),
        "placement": find_collection(client, "placement"),
        "budgets": find_collection(client, "budget"),
    }

    logger.info("Fetching data from collections...")
    reports = {k: fetch_collection_df(client, v) for k, v in collections_map.items()}

    campaign_name = campaign.lower()
    search_terms = reports["search_terms"]
    campaigns_df = reports["campaigns"]
    budgets = reports.get("budgets", pd.DataFrame())

    logger.info(f"Filtering for campaign: {campaign}")
    st_camp_col = find_campaign_column(search_terms)
    camp_camp_col = find_campaign_column(campaigns_df)

    search_terms = search_terms[search_terms[st_camp_col].str.lower() == campaign_name]
    campaigns_df = campaigns_df[campaigns_df[camp_camp_col].str.lower() == campaign_name]
    logger.info(f"  - Search terms: {len(search_terms)} rows")
    logger.info(f"  - Campaigns: {len(campaigns_df)} rows")

    if budgets is not None and not budgets.empty:
        try:
            budget_camp_col = find_campaign_column(budgets)
            budgets = budgets[budgets[budget_camp_col].str.lower() == campaign_name]
            logger.info(f"  - Budgets: {len(budgets)} rows")
        except KeyError:
            budgets = pd.DataFrame()

    if campaigns_df.empty:
        logger.error(f"Campaign not found: {campaign}")
        raise ValueError(f"Campaign not found: {campaign}")

    logger.info("Calculating date range...")
    start_date, end_date, num_days = calculate_date_range(campaigns_df)
    if start_date is None:
        start_date, end_date, num_days = calculate_date_range(search_terms)
    logger.info(f"  - Date range: {start_date or 'Unknown'} to {end_date or 'Unknown'} ({num_days} days)")

    logger.info("Computing metrics...")
    campaigns_df = normalize_metrics(campaigns_df)

    campaign_summary = campaigns_df.groupby(camp_camp_col, as_index=False).agg(
        spend=("spend", "sum"),
        sales=("sales", "sum"),
        clicks=("clicks", "sum"),
        impressions=("impressions", "sum"),
    )
    if "budget_amount" in campaigns_df.columns:
        campaign_summary["budget_amount"] = campaigns_df["budget_amount"].fillna(0).iloc[0] if len(campaigns_df) > 0 else 0
    
    # Convert totals to daily averages
    campaign_summary["spend_total"] = campaign_summary["spend"]
    campaign_summary["sales_total"] = campaign_summary["sales"]
    campaign_summary["clicks_total"] = campaign_summary["clicks"]
    campaign_summary["impressions_total"] = campaign_summary["impressions"]
    
    campaign_summary["spend"] = campaign_summary["spend"] / num_days
    campaign_summary["sales"] = campaign_summary["sales"] / num_days
    campaign_summary["clicks"] = campaign_summary["clicks"] / num_days
    campaign_summary["impressions"] = campaign_summary["impressions"] / num_days
    
    # Add date range info
    campaign_summary["date_range_start"] = start_date or "Unknown"
    campaign_summary["date_range_end"] = end_date or "Unknown"
    campaign_summary["num_days"] = num_days
    
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
        logger.info(f"  - Budget: ${budget_amt:.2f}")
    else:
        budget_amt = float(campaign_summary["budget_amount"].iloc[0]) if "budget_amount" in campaign_summary.columns else 0
        budget_info = {"total_budget": budget_amt, "daily_budget": budget_amt}
        logger.info(f"  - Budget: ${budget_amt:.2f}")

    logger.info("Analyzing keywords...")
    targeting_data = reports.get("targeting", pd.DataFrame())
    if targeting_data is not None and not targeting_data.empty:
        targeting_camp_col = find_campaign_column(targeting_data)
        targeting_data = targeting_data[targeting_data[targeting_camp_col].str.lower() == campaign_name]
        logger.info(f"  - Targeting rows: {len(targeting_data)}")
    else:
        targeting_data = pd.DataFrame()
    
    keyword_rollup, best_keywords, worst_keywords, keyword_types, alignment_analysis = compute_keyword_metrics(
        search_terms_df=search_terms,
        targeting_df=targeting_data,
        min_impr=min_impr,
        top_n=top_n,
    )
    logger.info(f"  - Keywords: {len(keyword_rollup)} total, {len(best_keywords)} best, {len(worst_keywords)} worst")
    logger.info(f"  - Well-aligned targeting: {len(alignment_analysis.get('well_aligned', []))}")
    logger.info(f"  - Untargeted customer searches: {len(alignment_analysis.get('untargeted_searches', []))}")

    payload = {
        "campaign_summary": campaign_summary.to_dict("records"),
        "budget": budget_info,
        "keyword_rollup": keyword_rollup,
        "best_keywords": best_keywords,
        "worst_keywords": worst_keywords,
        "keyword_types": keyword_types,
        "alignment_analysis": alignment_analysis,
    }

    logger.info(f"Calling OpenAI API (model: {model}, streaming: true)...")
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
    
    logger.info("Streaming response...")
    full_text = ""
    chunk_count = 0
    token_usage = None
    system_message = "You are a concise Amazon Ads optimization assistant."
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            full_text += content
            chunk_count += 1
            yield content
        
        # Check for token usage in chunk (usually in the last chunk)
        if hasattr(chunk, "usage") and chunk.usage:
            token_usage = extract_token_usage(chunk)
    
    logger.info(f"Stream complete: {chunk_count} chunks, {len(full_text)} chars")
    
    # Track token usage - estimate if not available from API
    tracker = get_tracker()
    if token_usage:
        tracker.record_usage(
            input_tokens=token_usage["input_tokens"],
            output_tokens=token_usage["output_tokens"],
            total_tokens=token_usage["total_tokens"],
            model=model,
            campaign=campaign,
            metadata={"function": "generate_recommendations_streaming", "streaming": True},
        )
    else:
        # Estimate tokens using tiktoken
        input_tokens = estimate_tokens(system_message + "\n\n" + prompt, model)
        output_tokens = estimate_tokens(full_text, model)
        total_tokens = input_tokens + output_tokens
        
        logger.info(f"Estimated token usage: {input_tokens:,} input + {output_tokens:,} output = {total_tokens:,} total")
        tracker.record_usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            model=model,
            campaign=campaign,
            metadata={"function": "generate_recommendations_streaming", "streaming": True, "estimated": True},
        )
    
    # Save to file after streaming completes
    logger.info("Saving recommendations to file...")
    output_file = save_recommendations_to_file(campaign, full_text)
    logger.info(f"Saved to: {output_file}")
    logger.info("=" * 60)
    logger.info("STREAMING COMPLETE")
    logger.info("=" * 60)
    
    # Final yield with metadata (caller can check for dict type)
    yield {"complete": True, "output_file": str(output_file), "full_text": full_text}


if __name__ == "__main__":
    main()
