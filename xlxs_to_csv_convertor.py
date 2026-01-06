import pandas as pd
import re
from pathlib import Path
from typing import Dict
import logging
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataConverter:
    """Convert and standardize Amazon Ads reports"""
    
    def __init__(self, output_directory: str = './csv_output'):
        self.output_dir = Path(output_directory)
        self.output_dir.mkdir(exist_ok=True)
    
    def convert_xlsx_to_csv(self, reports: Dict[str, pd.DataFrame]) -> Dict[str, Path]:
        """Convert all DataFrames to CSV files"""
        csv_files = {}
        
        for report_type, df in reports.items():
            # Generate CSV filename
            csv_filename = f"{report_type}_processed.csv"
            csv_path = self.output_dir / csv_filename
            
            # Save to CSV
            df.to_csv(csv_path, index=False, encoding='utf-8')
            csv_files[report_type] = csv_path
            
            logger.info(f"✓ Saved CSV: {csv_filename}")
        
        return csv_files
    
    def standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names (lowercase, underscores)"""
        df = df.copy()
        df.columns = [
            re.sub(r'\s+', '_', col.lower().strip())
            .replace('7_day_total_sales', 'sales')
            .replace('campaign_name', 'campaign')
            .replace('ad_group_name', 'ad_group')
            for col in df.columns
        ]
        return df
    
    def clean_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and convert numeric columns"""
        df = df.copy()
        
        # Columns that should be numeric
        numeric_cols = ['impressions', 'clicks', 'spend', 'sales', 'budget']
        
        for col in numeric_cols:
            if col in df.columns:
                # Remove currency symbols and convert
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace('$', '').str.replace(',', ''),
                    errors='coerce'
                ).fillna(0)
        
        return df
    
    def add_calculated_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calculated metrics (CTR, CPC, ACOS, ROAS)"""
        df = df.copy()
        
        # CTR (Click-Through Rate)
        if 'clicks' in df.columns and 'impressions' in df.columns:
            df['ctr'] = (df['clicks'] / df['impressions'].replace(0, 1)) * 100
        
        # CPC (Cost Per Click)
        if 'spend' in df.columns and 'clicks' in df.columns:
            df['cpc'] = df['spend'] / df['clicks'].replace(0, 1)
        
        # ACOS (Advertising Cost of Sales)
        if 'spend' in df.columns and 'sales' in df.columns:
            df['acos'] = (df['spend'] / df['sales'].replace(0, 1)) * 100
        
        # ROAS (Return on Ad Spend)
        if 'sales' in df.columns and 'spend' in df.columns:
            df['roas'] = df['sales'] / df['spend'].replace(0, 1)
        
        # Replace inf values with 0
        df = df.replace([float('inf'), float('-inf')], 0)
        
        return df
    
    def process_all_reports(self, reports: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Process all reports: standardize, clean, calculate metrics"""
        processed_reports = {}
        
        for report_type, df in reports.items():
            logger.info(f"\nProcessing {report_type}...")
            
            # Apply transformations
            df = self.standardize_column_names(df)
            df = self.clean_numeric_columns(df)
            df = self.add_calculated_metrics(df)
            
            # Remove rows with all zeros
            numeric_cols = df.select_dtypes(include=['number']).columns
            df = df[~(df[numeric_cols] == 0).all(axis=1)]
            
            processed_reports[report_type] = df
            
            logger.info(f"✓ Processed {report_type}: {len(df)} rows retained")
        
        return processed_reports


# Usage
if __name__ == "__main__":
    # Step 1: Load data
    loader = AmazonAdsDataLoader(data_directory='./amazon_reports')
    reports = loader.load_all_reports()
    
    # Step 2: Convert and process
    converter = DataConverter(output_directory='./csv_output')
    processed_reports = converter.process_all_reports(reports)
    csv_files = converter.convert_xlsx_to_csv(processed_reports)
    
    print("\n✅ All files converted and processed!")