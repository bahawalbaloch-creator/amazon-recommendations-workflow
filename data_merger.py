import pandas as pd
from typing import Dict, List
from amazon_data_loader import AmazonAdsDataLoader
from xlxs_to_csv_convertor import DataConverter
import logging
from pathlib import Path
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataMerger:
    """Merge and aggregate Amazon Ads reports"""
    
    def __init__(self, reports: Dict[str, pd.DataFrame]):
        self.reports = reports
        self._ensure_numeric_columns()
    
    def _ensure_numeric_columns(self):
        """Ensure all numeric columns are float type"""
        numeric_cols = ['impressions', 'clicks', 'spend', 'sales', 'budget', 'budget_amount']
        
        for report_name, df in self.reports.items():
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)
            logger.info(f"Ensured numeric types for {report_name}")
        """Create comprehensive campaign-level summary"""
        
        # Start with campaign report
        campaign_df = self.reports['campaigns'].copy()
        
        # Add budget data
        if 'budgets' in self.reports:
            budget_df = self.reports['budgets'][['campaign', 'budget_amount']].copy()
            campaign_df = campaign_df.merge(
                budget_df, 
                on='campaign', 
                how='left',
                suffixes=('', '_budget')
            )
        
        # Calculate campaign metrics
        campaign_df['spend_to_budget_ratio'] = \
            campaign_df['spend'] / campaign_df['budget_amount'].replace(0, 1)
        
        # Sort by spend
        campaign_df = campaign_df.sort_values('spend', ascending=False)
        
        return campaign_df

    def create_campaign_summary(self) -> pd.DataFrame:
        """Create comprehensive campaign-level summary"""
        
        # Start with campaign report
        campaign_df = self.reports['campaigns'].copy()
        
        # Add budget data
        if 'budgets' in self.reports:
            budget_df = self.reports['budgets'][['campaign', 'budget_amount']].copy()
            campaign_df = campaign_df.merge(
                budget_df, 
                on='campaign', 
                how='left',
                suffixes=('', '_budget')
            )
        
        # Calculate campaign metrics
        campaign_df['spend_to_budget_ratio'] = \
            campaign_df['spend'] / campaign_df['budget_amount'].replace(0, 1)
        
        # Sort by spend
        campaign_df = campaign_df.sort_values('spend', ascending=False)
        
        return campaign_df
    
    def create_keyword_performance(self) -> pd.DataFrame:
        """Aggregate search terms with performance metrics"""
        
        search_df = self.reports['search_terms'].copy()
        
        # Group by search term across all campaigns
        keyword_agg = search_df.groupby('customer_search_term').agg({
            'impressions': 'sum',
            'clicks': 'sum',
            'spend': 'sum',
            'sales': 'sum',
            'campaign': 'nunique'  # Number of campaigns using this keyword
        }).reset_index()
        
        # Recalculate metrics
        keyword_agg['ctr'] = (keyword_agg['clicks'] / keyword_agg['impressions'].replace(0, 1)) * 100
        keyword_agg['cpc'] = keyword_agg['spend'] / keyword_agg['clicks'].replace(0, 1)
        keyword_agg['acos'] = (keyword_agg['spend'] / keyword_agg['sales'].replace(0, 1)) * 100
        keyword_agg['roas'] = keyword_agg['sales'] / keyword_agg['spend'].replace(0, 1)
        
        keyword_agg = keyword_agg.replace([float('inf'), float('-inf')], 0)
        
        # Sort by sales
        keyword_agg = keyword_agg.sort_values('sales', ascending=False)
        
        return keyword_agg
    
    def create_placement_analysis(self) -> pd.DataFrame:
        """Analyze performance by placement type"""
        
        placement_df = self.reports['placement'].copy()
        
        # Group by placement
        placement_agg = placement_df.groupby('placement').agg({
            'impressions': 'sum',
            'clicks': 'sum',
            'spend': 'sum',
            'sales': 'sum',
            'campaign': 'nunique'
        }).reset_index()
        
        # Calculate metrics
        placement_agg['ctr'] = (placement_agg['clicks'] / placement_agg['impressions'].replace(0, 1)) * 100
        placement_agg['acos'] = (placement_agg['spend'] / placement_agg['sales'].replace(0, 1)) * 100
        placement_agg['roas'] = placement_agg['sales'] / placement_agg['spend'].replace(0, 1)
        placement_agg['spend_share'] = (placement_agg['spend'] / placement_agg['spend'].sum()) * 100
        
        placement_agg = placement_agg.replace([float('inf'), float('-inf')], 0)
        
        return placement_agg
    
    def create_targeting_insights(self) -> pd.DataFrame:
        """Analyze targeting performance"""
        
        targeting_df = self.reports['targeting'].copy()
        
        # Group by campaign and ad group
        targeting_agg = targeting_df.groupby(['campaign', 'ad_group', 'targeting']).agg({
            'impressions': 'sum',
            'clicks': 'sum',
            'spend': 'sum',
            'sales': 'sum'
        }).reset_index()
        
        # Calculate metrics
        targeting_agg['acos'] = (targeting_agg['spend'] / targeting_agg['sales'].replace(0, 1)) * 100
        targeting_agg['roas'] = targeting_agg['sales'] / targeting_agg['spend'].replace(0, 1)
        targeting_agg['cpc'] = targeting_agg['spend'] / targeting_agg['clicks'].replace(0, 1)
        
        targeting_agg = targeting_agg.replace([float('inf'), float('-inf')], 0)
        
        # Flag underperformers (high ACOS, low ROAS)
        targeting_agg['underperformer'] = (
            (targeting_agg['acos'] > 50) & 
            (targeting_agg['spend'] > 10) &
            (targeting_agg['roas'] < 1.5)
        )
        
        return targeting_agg
    
    def create_master_dataset(self) -> Dict[str, pd.DataFrame]:
        """Create all aggregated datasets"""
        
        master_data = {
            'campaign_summary': self.create_campaign_summary(),
            'keyword_performance': self.create_keyword_performance(),
            'placement_analysis': self.create_placement_analysis(),
            'targeting_insights': self.create_targeting_insights()
        }
        
        # Log summaries
        for name, df in master_data.items():
            logger.info(f"✓ Created {name}: {len(df)} rows")
        
        return master_data


# Usage
if __name__ == "__main__":
    # Steps 1-2: Load and process
    loader = AmazonAdsDataLoader(data_directory='./amazon_reports')
    reports = loader.load_all_reports()
    
    converter = DataConverter()
    processed_reports = converter.process_all_reports(reports)
    
    # Step 3: Merge and aggregate
    merger = DataMerger(processed_reports)
    master_data = merger.create_master_dataset()
    
    # Save aggregated data
    output_dir = Path('./aggregated_data')
    output_dir.mkdir(exist_ok=True)
    
    for name, df in master_data.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)
        print(f"✓ Saved {name}.csv")
    
    print("\n✅ Data merging complete!")