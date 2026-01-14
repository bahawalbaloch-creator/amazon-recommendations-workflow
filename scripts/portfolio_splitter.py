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


class PortfolioSplitter:
    """Split each data file by portfolio, creating separate files for each portfolio"""
    
    def __init__(self, data_directory: str = './data', output_directory: str = './portfolio_split_output'):
        self.data_directory = Path(data_directory)
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(exist_ok=True)
    
    def find_data_files(self) -> List[Path]:
        """Find all data files (CSV and XLSX) in the data directory"""
        all_files = list(self.data_directory.glob('*.xlsx')) + \
                   list(self.data_directory.glob('*.csv'))
        
        logger.info(f"Found {len(all_files)} files in {self.data_directory}")
        return all_files
    
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
                logger.warning(f"File is empty: {file_path.name}")
                return pd.DataFrame()
            
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
    
    def extract_portfolio_name(self, row: pd.Series, portfolio_column: Optional[str] = None) -> str:
        """
        Extract portfolio name from row data.
        First checks for portfolio column, then tries to extract from campaign name.
        """
        # Check if there's a portfolio column
        if portfolio_column and portfolio_column in row.index:
            portfolio_value = row[portfolio_column]
            if pd.notna(portfolio_value) and str(portfolio_value).strip():
                return str(portfolio_value).strip()
        
        # Try to extract from campaign name
        campaign_cols = [col for col in row.index if 'campaign' in col.lower()]
        if campaign_cols:
            campaign_name = row[campaign_cols[0]]
            if pd.notna(campaign_name):
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
                
                # If no separator, use first word or full campaign name
                if campaign_str:
                    return campaign_str.split()[0] if campaign_str.split() else campaign_str
        
        return 'Unknown Portfolio'
    
    def identify_portfolio_column(self, df: pd.DataFrame) -> Optional[str]:
        """Identify if there's a portfolio column in the dataframe"""
        portfolio_cols = [col for col in df.columns if 'portfolio' in col.lower()]
        if portfolio_cols:
            return portfolio_cols[0]
        return None
    
    def split_file_by_portfolio(self, file_path: Path) -> List[Path]:
        """
        Split a single file by portfolio and save separate files.
        Returns list of created file paths.
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing file: {file_path.name}")
        logger.info(f"{'='*60}")
        
        # Load file
        df = self.load_file(file_path)
        
        if df.empty:
            logger.warning(f"Skipping empty file: {file_path.name}")
            return []
        
        # Standardize column names
        df = self.standardize_column_names(df)
        
        # Identify portfolio column
        portfolio_column = self.identify_portfolio_column(df)
        
        if portfolio_column:
            logger.info(f"Found portfolio column: {portfolio_column}")
        else:
            logger.info("No portfolio column found, will extract from campaign names")
        
        # Extract portfolio for each row
        df['_portfolio'] = df.apply(
            lambda row: self.extract_portfolio_name(row, portfolio_column),
            axis=1
        )
        
        # Get unique portfolios
        unique_portfolios = df['_portfolio'].unique()
        logger.info(f"Found {len(unique_portfolios)} portfolios: {', '.join(unique_portfolios[:5])}{'...' if len(unique_portfolios) > 5 else ''}")
        
        # Create base filename (without extension)
        base_filename = file_path.stem
        file_extension = file_path.suffix
        
        # Split by portfolio and save
        created_files = []
        for portfolio in unique_portfolios:
            # Filter data for this portfolio
            portfolio_df = df[df['_portfolio'] == portfolio].copy()
            
            # Remove the temporary portfolio column
            portfolio_df = portfolio_df.drop(columns=['_portfolio'])
            
            # Create safe portfolio name for filename
            safe_portfolio_name = re.sub(r'[^\w\s-]', '', str(portfolio)).strip()
            safe_portfolio_name = re.sub(r'[-\s]+', '_', safe_portfolio_name)
            
            # Create output filename: {portfolio_name}_{original_filename}
            output_filename = f"{safe_portfolio_name}_{base_filename}{file_extension}"
            output_path = self.output_directory / output_filename
            
            # Save to CSV (always save as CSV for consistency)
            if file_extension == '.xlsx':
                output_path = self.output_directory / f"{safe_portfolio_name}_{base_filename}.csv"
            
            portfolio_df.to_csv(output_path, index=False, encoding='utf-8')
            
            logger.info(f"  ✓ Created: {output_filename} ({len(portfolio_df)} rows)")
            created_files.append(output_path)
        
        return created_files
    
    def process_all_files(self):
        """Process all files in the data directory and split by portfolio"""
        logger.info("="*60)
        logger.info("Starting Portfolio Splitter")
        logger.info("="*60)
        
        # Find all data files
        data_files = self.find_data_files()
        
        if not data_files:
            logger.error(f"No data files found in {self.data_directory}")
            return
        
        # Process each file
        all_created_files = []
        for file_path in data_files:
            try:
                created_files = self.split_file_by_portfolio(file_path)
                all_created_files.extend(created_files)
            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {str(e)}")
                continue
        
        logger.info(f"\n{'='*60}")
        logger.info("✅ Portfolio splitting complete!")
        logger.info(f"Created {len(all_created_files)} portfolio-specific files")
        logger.info(f"Output directory: {self.output_directory}")
        logger.info(f"{'='*60}")
        
        return all_created_files


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Split Amazon Ads data files by portfolio',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python portfolio_splitter.py
  python portfolio_splitter.py --data-dir ./data --output-dir ./portfolio_split_output
  
This script splits each file in the data directory by portfolio.
For example, if Campaign_L30.csv has 10 portfolios, it will create:
  - portfolio1_Campaign_L30.csv
  - portfolio2_Campaign_L30.csv
  - ...
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
        default='./portfolio_split_output',
        help='Output directory for split files (default: ./portfolio_split_output)'
    )
    
    args = parser.parse_args()
    
    # Create splitter and process files
    splitter = PortfolioSplitter(
        data_directory=args.data_dir,
        output_directory=args.output_dir
    )
    
    splitter.process_all_files()


if __name__ == "__main__":
    main()
