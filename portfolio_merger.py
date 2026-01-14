import pandas as pd
import re
from pathlib import Path
from typing import Dict, List, Optional
import logging
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PortfolioMerger:
    """Merge all Amazon Ads data files and organize by portfolio"""
    
    def __init__(self, data_directory: str = './data', output_directory: str = './portfolio_output'):
        self.data_directory = Path(data_directory)
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(exist_ok=True)
        
        # File patterns to match
        self.file_patterns = {
            'search_terms': 'search_term',
            'campaigns': 'campaign',
            'targeting': 'targeting',
            'placement': 'placement',
            'budgets': 'budget'
        }
    
    def find_files(self) -> Dict[str, Path]:
        """Find all report files in directory"""
        found_files = {}
        
        # Get all xlsx and csv files
        all_files = list(self.data_directory.glob('*.xlsx')) + \
                   list(self.data_directory.glob('*.csv'))
        
        logger.info(f"Found {len(all_files)} files in {self.data_directory}")
        
        for file_path in all_files:
            file_name = file_path.stem.lower()
            
            # Match file to report type
            for report_type, pattern in self.file_patterns.items():
                if pattern in file_name:
                    if report_type in found_files:
                        logger.warning(f"Duplicate file for {report_type}: {file_path.name}")
                    found_files[report_type] = file_path
                    logger.info(f"✓ Matched {report_type}: {file_path.name}")
                    break
        
        return found_files
    
    def load_file(self, file_path: Path) -> pd.DataFrame:
        """Load a single file (XLSX or CSV)"""
        try:
            if file_path.suffix == '.xlsx':
                df = pd.read_excel(file_path, engine='openpyxl')
                logger.info(f"Loaded XLSX: {file_path.name} ({len(df)} rows)")
            elif file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
                logger.info(f"Loaded CSV: {file_path.name} ({len(df)} rows)")
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            if df.empty:
                raise ValueError(f"File is empty: {file_path.name}")
            
            # Strip whitespace from column names
            df.columns = df.columns.str.strip()
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading {file_path.name}: {str(e)}")
            raise
    
    def standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names (lowercase, underscores)"""
        df = df.copy()
        df.columns = [
            re.sub(r'\s+', '_', col.lower().strip())
            .replace('7_day_total_sales', 'sales')
            .replace('campaign_name', 'campaign')
            .replace('ad_group_name', 'ad_group')
            .replace('customer_search_term', 'search_term')
            .replace('budget_amount', 'budget')
            for col in df.columns
        ]
        return df
    
    def clean_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and convert numeric columns"""
        df = df.copy()
        
        # Columns that should be numeric
        numeric_cols = ['impressions', 'clicks', 'spend', 'sales', 'budget', 'budget_amount']
        
        for col in numeric_cols:
            if col in df.columns:
                # Remove currency symbols and convert
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace('$', '').str.replace(',', ''),
                    errors='coerce'
                ).fillna(0)
        
        return df
    
    def extract_portfolio_name(self, campaign_name: str, portfolio_column: Optional[str] = None) -> str:
        """
        Extract portfolio name from campaign name or use portfolio column if available.
        Common patterns:
        - "Portfolio Name - Campaign Name"
        - "Portfolio/Campaign"
        - "Portfolio | Campaign"
        """
        if pd.isna(campaign_name) or campaign_name == '':
            return 'Unknown Portfolio'
        
        campaign_str = str(campaign_name).strip()
        
        # Try common separators
        separators = [' - ', ' / ', ' | ', ' :: ', ' ::', ':: ']
        for sep in separators:
            if sep in campaign_str:
                parts = campaign_str.split(sep, 1)
                if len(parts) == 2:
                    portfolio = parts[0].strip()
                    if portfolio:
                        return portfolio
        
        # If no separator found, check if it starts with a known portfolio pattern
        # For now, if no clear portfolio, use the campaign name as portfolio
        # (user can adjust this logic based on their naming convention)
        return campaign_str.split()[0] if campaign_str else 'Unknown Portfolio'
    
    def identify_portfolios(self, reports: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
        """
        Identify all portfolios and their associated campaigns.
        Returns dict: {portfolio_name: [list of campaign names]}
        """
        portfolio_campaigns = defaultdict(set)
        
        # Check if there's a portfolio column in any report
        portfolio_column = None
        for report_type, df in reports.items():
            possible_portfolio_cols = [col for col in df.columns if 'portfolio' in col.lower()]
            if possible_portfolio_cols:
                portfolio_column = possible_portfolio_cols[0]
                logger.info(f"Found portfolio column: {portfolio_column} in {report_type}")
                break
        
        # Get all unique campaigns
        all_campaigns = set()
        for report_type, df in reports.items():
            if 'campaign' in df.columns:
                campaigns = df['campaign'].dropna().unique()
                all_campaigns.update(campaigns)
        
        logger.info(f"Found {len(all_campaigns)} unique campaigns")
        
        # Extract portfolio for each campaign
        for campaign in all_campaigns:
            if portfolio_column:
                # Use portfolio column if available
                for report_type, df in reports.items():
                    if portfolio_column in df.columns and 'campaign' in df.columns:
                        portfolio_rows = df[df['campaign'] == campaign][portfolio_column].dropna()
                        if len(portfolio_rows) > 0:
                            portfolio = str(portfolio_rows.iloc[0]).strip()
                            if portfolio:
                                portfolio_campaigns[portfolio].add(campaign)
                                break
            else:
                # Extract from campaign name
                portfolio = self.extract_portfolio_name(campaign)
                portfolio_campaigns[portfolio].add(campaign)
        
        # Convert sets to lists
        result = {port: sorted(list(campaigns)) for port, campaigns in portfolio_campaigns.items()}
        
        logger.info(f"Identified {len(result)} portfolios:")
        for port, campaigns in result.items():
            logger.info(f"  - {port}: {len(campaigns)} campaigns")
        
        return result
    
    def filter_by_campaigns(self, df: pd.DataFrame, campaigns: List[str], campaign_col: str = 'campaign') -> pd.DataFrame:
        """Filter dataframe to only include specified campaigns"""
        if campaign_col not in df.columns:
            return pd.DataFrame()  # Return empty if no campaign column
        
        return df[df[campaign_col].isin(campaigns)].copy()
    
    def merge_portfolio_data(self, reports: Dict[str, pd.DataFrame], portfolio: str, campaigns: List[str]) -> pd.DataFrame:
        """
        Merge all data for a specific portfolio into one comprehensive dataframe.
        Creates a wide format with all information, with each row representing a data point
        (campaign, keyword, placement, etc.) with its associated metrics.
        """
        portfolio_data = []
        
        # 1. Campaign-level data
        if 'campaigns' in reports:
            campaign_df = self.filter_by_campaigns(reports['campaigns'], campaigns)
            if not campaign_df.empty:
                campaign_df['data_type'] = 'campaign'
                campaign_df['record_id'] = campaign_df['campaign']
                portfolio_data.append(campaign_df)
        
        # 2. Budget data
        if 'budgets' in reports:
            budget_df = self.filter_by_campaigns(reports['budgets'], campaigns)
            if not budget_df.empty:
                budget_df['data_type'] = 'budget'
                budget_df['record_id'] = budget_df['campaign']
                portfolio_data.append(budget_df)
        
        # 3. Search terms (keywords)
        if 'search_terms' in reports:
            search_df = self.filter_by_campaigns(reports['search_terms'], campaigns)
            if not search_df.empty:
                search_df['data_type'] = 'keyword'
                # Create unique ID for each keyword-campaign combination
                if 'search_term' in search_df.columns:
                    search_df['record_id'] = search_df['campaign'].astype(str) + ' | ' + search_df['search_term'].astype(str)
                else:
                    search_df['record_id'] = search_df['campaign'].astype(str)
                portfolio_data.append(search_df)
        
        # 4. Placement data
        if 'placement' in reports:
            placement_df = self.filter_by_campaigns(reports['placement'], campaigns)
            if not placement_df.empty:
                placement_df['data_type'] = 'placement'
                if 'placement' in placement_df.columns:
                    placement_df['record_id'] = placement_df['campaign'].astype(str) + ' | ' + placement_df['placement'].astype(str)
                else:
                    placement_df['record_id'] = placement_df['campaign'].astype(str)
                portfolio_data.append(placement_df)
        
        # 5. Targeting data
        if 'targeting' in reports:
            targeting_df = self.filter_by_campaigns(reports['targeting'], campaigns)
            if not targeting_df.empty:
                targeting_df['data_type'] = 'targeting'
                if 'targeting' in targeting_df.columns:
                    targeting_df['record_id'] = targeting_df['campaign'].astype(str) + ' | ' + targeting_df['targeting'].astype(str)
                else:
                    targeting_df['record_id'] = targeting_df['campaign'].astype(str)
                portfolio_data.append(targeting_df)
        
        # Combine all data
        if not portfolio_data:
            logger.warning(f"No data found for portfolio: {portfolio}")
            return pd.DataFrame()
        
        # Concatenate all dataframes - each row represents a different data point
        # (campaign summary, keyword, placement, etc.)
        merged_df = pd.concat(portfolio_data, ignore_index=True, sort=False)
        
        # Add portfolio column
        merged_df['portfolio'] = portfolio
        
        # Reorder columns to put portfolio and data_type first
        priority_cols = ['portfolio', 'data_type', 'campaign', 'record_id']
        other_cols = [c for c in merged_df.columns if c not in priority_cols]
        cols = [c for c in priority_cols if c in merged_df.columns] + other_cols
        merged_df = merged_df[cols]
        
        # Sort by data_type and then campaign for better readability
        if 'data_type' in merged_df.columns and 'campaign' in merged_df.columns:
            merged_df = merged_df.sort_values(['data_type', 'campaign'], na_position='last')
        
        return merged_df
    
    def process_all_reports(self) -> Dict[str, pd.DataFrame]:
        """Load and process all reports"""
        files = self.find_files()
        reports = {}
        
        for report_type, file_path in files.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {report_type}: {file_path.name}")
            logger.info(f"{'='*60}")
            
            # Load file
            df = self.load_file(file_path)
            
            # Standardize and clean
            df = self.standardize_column_names(df)
            df = self.clean_numeric_columns(df)
            
            reports[report_type] = df
            
            logger.info(f"✓ Processed {report_type}: {len(df)} rows")
        
        return reports
    
    def generate_portfolio_csvs(self):
        """Main method to generate CSV files for each portfolio"""
        logger.info("="*60)
        logger.info("Starting Portfolio Merger")
        logger.info("="*60)
        
        # Load and process all reports
        reports = self.process_all_reports()
        
        # Identify portfolios
        portfolios = self.identify_portfolios(reports)
        
        if not portfolios:
            logger.error("No portfolios identified. Please check your data.")
            return
        
        # Generate CSV for each portfolio
        generated_files = []
        for portfolio, campaigns in portfolios.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing Portfolio: {portfolio}")
            logger.info(f"Campaigns: {', '.join(campaigns[:5])}{'...' if len(campaigns) > 5 else ''}")
            logger.info(f"{'='*60}")
            
            # Merge all data for this portfolio
            portfolio_df = self.merge_portfolio_data(reports, portfolio, campaigns)
            
            if portfolio_df.empty:
                logger.warning(f"No data to export for portfolio: {portfolio}")
                continue
            
            # Create safe filename
            safe_portfolio_name = re.sub(r'[^\w\s-]', '', portfolio).strip()
            safe_portfolio_name = re.sub(r'[-\s]+', '_', safe_portfolio_name)
            filename = f"portfolio_{safe_portfolio_name}.csv"
            filepath = self.output_directory / filename
            
            # Save to CSV
            portfolio_df.drop(columns=['program_type', 'retailer', 'country', 'currency'], inplace=True)
            portfolio_df.to_csv(filepath, index=False, encoding='utf-8')
            
            logger.info(f"✓ Saved: {filename}")
            logger.info(f"  - Rows: {len(portfolio_df)}")
            logger.info(f"  - Columns: {len(portfolio_df.columns)}")
            logger.info(f"  - Data types: {portfolio_df['data_type'].value_counts().to_dict()}")
            
            generated_files.append(filepath)
        
        logger.info(f"\n{'='*60}")
        logger.info("✅ Portfolio merge complete!")
        logger.info(f"Generated {len(generated_files)} portfolio CSV files")
        logger.info(f"Output directory: {self.output_directory}")
        logger.info(f"{'='*60}")
        
        return generated_files


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Merge Amazon Ads data files by portfolio',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python portfolio_merger.py
  python portfolio_merger.py --data-dir ./data --output-dir ./portfolio_output
        """
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data',
        help='Directory containing Amazon Ads report files (default: ./data)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./portfolio_output',
        help='Output directory for portfolio CSV files (default: ./portfolio_output)'
    )
    
    args = parser.parse_args()
    
    # Create merger and generate files
    merger = PortfolioMerger(
        data_directory=args.data_dir,
        output_directory=args.output_dir
    )
    
    merger.generate_portfolio_csvs()


if __name__ == "__main__":
    main()
