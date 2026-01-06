import json
import openai
from typing import Dict
import pandas as pd
from amazon_data_loader import AmazonAdsDataLoader
from xlxs_to_csv_convertor import DataConverter
import logging
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMOptimizer:
    """Prepare data and call GPT-4o for optimization recommendations"""
    
    def __init__(self, api_key: str, campaign_filter: str = None):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"
        self.campaign_filter = campaign_filter
    
    # def prepare_data_for_llm(self, reports: Dict[str, pd.DataFrame]) -> Dict:
    #     """Convert DataFrames to JSON-serializable format"""
        
    #     # Limit data size to avoid token limits
    #     data_payload = {
    #         'search_terms': reports['search_terms'].head(100).to_dict('records'),
    #         'campaigns': reports['campaigns'].to_dict('records'),
    #         'targeting': reports['targeting'].head(100).to_dict('records'),
    #         'placement': reports['placement'].to_dict('records'),
    #         'budgets': reports['budgets'].to_dict('records')
    #     }
        
    #     # Convert NaN to None for JSON serialization
    #     data_payload = json.loads(
    #         json.dumps(data_payload, default=str, allow_nan=True)
    #     )
        
    #     return data_payload
    
    def build_system_prompt(self) -> str:
        return """
            You are an Amazon Sponsored Products optimization expert.

            You will receive:
            - campaign_summary
            - search_term_winners
            - search_term_losers
            - placement_summary

            Tasks:
            1. Recommend campaign bid, placement, and budget changes.
            2. Recommend exact-match keywords from winners.
            3. Recommend negative keywords from losers.
            4. Recommend targets to pause or increase bids.

            Return ONLY valid JSON in this format:

            {
            "campaign_adjustments": [],
            "keyword_recommendations": {
                "add_exact": [],
                "negative": []
            },
            "targeting_recommendations": [],
            "Detailed reasoning": ""
            }
            
            """

    def prepare_data_for_llm(self, reports: Dict[str, pd.DataFrame]) -> Dict:
        """
        Aggregate Amazon Ads data into LLM-ready features
        """

        # --- Campaign-level aggregation ---
        campaigns = reports['campaigns']
        print("campaigns.columns:", campaigns.columns)
        campaign_summary = (
            campaigns
            .groupby('campaign', as_index=False)
            .agg(
                spend=('spend', 'sum'),
                sales=('sales', 'sum'),
                clicks=('clicks', 'sum'),
                impressions=('impressions', 'sum'),
                orders=('7_day_total_orders_(#)', 'sum'),
                budget=('budget_amount', 'first')
            )
        )

        campaign_summary['acos'] = (
            campaign_summary['spend'] / campaign_summary['sales']
        ).replace([float('inf')], None)

        campaign_summary['cvr'] = (
            campaign_summary['orders'] / campaign_summary['clicks']
        ).replace([float('inf')], None)

        # Keep only meaningful campaigns
        campaign_summary = campaign_summary[
            (campaign_summary['spend'] > 10) & 
            (campaign_summary['acos'] > 0.5)
        ]
        
        # Filter to specific campaign if requested
        if self.campaign_filter:
            campaign_summary = campaign_summary[
                campaign_summary['campaign'].str.lower() == self.campaign_filter.lower()
            ]
        
        campaign_summary = campaign_summary.sort_values('acos', ascending=False).head(5)

        # --- Search terms: winners & losers ---
        search_terms = reports['search_terms']
        print("search_terms columns:", search_terms.columns)

        search_terms_summary = (
            search_terms
            .groupby(['campaign', 'ad_group', 'customer_search_term'], as_index=False)
            .agg(
                spend=('spend', 'sum'),
                sales=('sales', 'sum'),
                orders=('7_day_total_orders_(#)', 'sum'),
                clicks=('clicks', 'sum'),
                impressions=('impressions', 'sum')
            )
        )

        search_terms_summary['ctr'] = (
            search_terms_summary['clicks'] / search_terms_summary['impressions']
        ).replace([float('inf')], 0).fillna(0)

        search_terms_summary['acos'] = (
            search_terms_summary['spend'] / search_terms_summary['sales']
        ).replace([float('inf')], None)

        winners = (
            search_terms_summary
            .query("impressions > 100")
            .sort_values(['impressions', 'ctr'], ascending=[False, False])
            .head(5)
        )

        losers = (
            search_terms_summary
            .query("impressions > 100")
            .sort_values('ctr', ascending=True)
            .head(5)
        )

        # --- Placement performance ---
        placement = reports['placement']
        print("placement columns:", placement.columns)
        placement_summary = (
            placement
            .groupby(['campaign', 'placement'], as_index=False)
            .agg(
                spend=('spend', 'sum'),
                sales=('sales', 'sum')
            )
        )

        placement_summary['acos'] = (
            placement_summary['spend'] / placement_summary['sales']
        ).replace([float('inf')], None)

        # --- Final payload ---
        payload = {
            "campaign_summary": campaign_summary.to_dict('records'),
            "search_term_winners": winners.to_dict('records'),
            "search_term_losers": losers.to_dict('records'),
            "placement_summary": placement_summary.to_dict('records'),
        }

        return json.loads(json.dumps(payload, default=str))

    # def build_system_prompt(self) -> str:
    #     """Build the system prompt for GPT-4o"""
        
    #     return """You are an Amazon Ads Optimization Assistant. You will receive five structured datasets from Sponsored Products reports:

    #             - search_terms
    #             - campaigns
    #             - targeting
    #             - placement
    #             - budgets

    #             Your goal is to generate precise performance recommendations for bid strategy, targeting, and budget scaling.

    #             ---

    #             1. Campaign Adjustments:
    #             For each campaign, return:
    #             - campaign_name (string)
    #             - default_bid_multiplier (float, optional — only if bid should change)
    #             - bid_adjustments: { top_of_search, rest_of_search, product_pages } (percentages)
    #             - budget_change: { action: increase | decrease | none, percent: float }
    #             - projected_daily_spend_usd (float)
    #             - projected_daily_sales_usd (float)
    #             - estimated_acos_percent (float)
    #             - estimated_roas_multiple (float)

    #             Base projections on historical 30-day data. If a budget increase is recommended, scale projected spend and sales proportionally. Return NaN only if data is insufficient.

    #             ---

    #             2. Keyword Recommendations:
    #             Recommend at least 5 exact-match keywords to add. Each must include:
    #             - term
    #             - campaign_name
    #             - ad_group_name
    #             - suggested_bid (USD)

    #             Also return at least 3 negative keywords:
    #             - { term: "...", campaign_name?: "..." }

    #             Do not return keyword recommendations that lack campaign and ad group names.

    #             ---

    #             3. Targeting Recommendations:
    #             Recommend at least 3 targets to pause or increase bids. Return:
    #             - target (ASIN, keyword, or match group)
    #             - campaign_name
    #             - ad_group_name
    #             - action: "pause" or "increase_bid"
    #             - value: float (if increasing bid)

    #             ---

    #             Respond ONLY with a JSON object in this exact format. Do NOT include backticks, code blocks, or explanations:

    #             {
    #             "campaign_adjustments": [...],
    #             "keyword_recommendations": {
    #                 "add_exact": [...],
    #                 "negative": [...]
    #             },
    #             "targeting_recommendations": [...]
    #             }"""
                    
    def call_gpt4o(self, data_payload: Dict) -> Dict:
        """Make API call to GPT-4o"""
        
        system_prompt = self.build_system_prompt()
        user_message = f"Here is the Amazon Ads data:\n\n{json.dumps(data_payload, indent=2)}"
        
        logger.info("Making API call to GPT-4o...")
        logger.info(f"Data payload size: {len(json.dumps(data_payload))} characters")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,
                max_tokens=4000,
                response_format={"type": "json_object"}
            )
            
            # Extract response
            raw_response = response.choices[0].message.content
            logger.info(f"✓ Received response ({len(raw_response)} characters)")
            
            # Parse JSON
            recommendations = json.loads(raw_response)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            raise
    
    def validate_recommendations(self, recommendations: Dict) -> bool:
        """Validate the structure of recommendations"""
        
        required_keys = [
            'campaign_adjustments',
            'keyword_recommendations',
            'targeting_recommendations'
        ]
        
        for key in required_keys:
            if key not in recommendations:
                logger.error(f"Missing required key: {key}")
                return False
        
        # Validate keyword recommendations structure
        if 'add_exact' not in recommendations['keyword_recommendations']:
            logger.error("Missing 'add_exact' in keyword_recommendations")
            return False
        
        if 'negative' not in recommendations['keyword_recommendations']:
            logger.error("Missing 'negative' in keyword_recommendations")
            return False
        
        logger.info("✓ Recommendations structure validated")
        return True


# Usage
if __name__ == "__main__":
    # Steps 1-3: Load, process, and merge data
    loader = AmazonAdsDataLoader(data_directory='./amazon_reports')
    reports = loader.load_all_reports()
    
    converter = DataConverter()
    processed_reports = converter.process_all_reports(reports)
    
    # Step 4: Call LLM
    optimizer = LLMOptimizer(api_key='your-openai-api-key')
    
    # Prepare data
    data_payload = optimizer.prepare_data_for_llm(processed_reports)

    
    # Make API call
    recommendations = optimizer.call_gpt4o(data_payload)
    
    # Validate
    if optimizer.validate_recommendations(recommendations):
        print("\n✅ Recommendations generated successfully!")
    
    # Save recommendations
    with open('./recommendations.json', 'w') as f:
        json.dump(recommendations, f, indent=2)