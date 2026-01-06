import pandas as pd
import os
from pathlib import Path
from typing import Dict, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AmazonAdsDataLoader:
    """Handles loading and validation of Amazon Ads reports"""
    
    def __init__(self, data_directory: str):
        self.data_directory = Path(data_directory)
        
        # Define expected file patterns and required columns
        self.file_patterns = {
            'search_terms': 'search_term',
            'campaigns': 'campaign',
            'targeting': 'targeting',
            'placement': 'placement',
            'budgets': 'budget'
        }
        
        # Required columns for each report type
        self.required_columns = {
            'search_terms': ['Customer Search Term', 'Campaign Name', 'Ad Group Name', 
                           'Impressions', 'Clicks', 'Spend', '7 Day Total Sales'],
            'campaigns': ['Campaign Name', 'Impressions', 'Clicks', 
                         'Spend', '7 Day Total Sales', 'Budget Amount'],
            'targeting': ['Targeting', 'Campaign Name', 'Ad Group Name', 
                         'Impressions', 'Clicks', 'Spend', '7 Day Total Sales'],
            'placement': ['Campaign Name', 'Placement', 'Impressions', 
                         'Clicks', 'Spend', '7 Day Total Sales'],
            'budgets': ['Campaign Name', 'Budget Amount', 'Spend']
        }
    
    def find_files(self) -> Dict[str, Path]:
        """Find all required report files in directory"""
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
        
        # Check for missing files
        missing = set(self.file_patterns.keys()) - set(found_files.keys())
        if missing:
            raise FileNotFoundError(f"Missing required reports: {missing}")
        
        return found_files
    
    def validate_columns(self, df: pd.DataFrame, report_type: str) -> bool:
        """Validate that DataFrame has required columns"""
        required = self.required_columns[report_type]
        missing = set(required) - set(df.columns)
        
        if missing:
            logger.error(f"Missing columns in {report_type}: {missing}")
            logger.info(f"Available columns: {df.columns.tolist()}")
            return False
        
        return True
    
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
            
            # Basic validation
            if df.empty:
                raise ValueError(f"File is empty: {file_path.name}")
            
            # Strip whitespace from column names
            df.columns = df.columns.str.strip()
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading {file_path.name}: {str(e)}")
            raise
    
    def load_all_reports(self) -> Dict[str, pd.DataFrame]:
        """Load and validate all reports"""
        files = self.find_files()
        reports = {}
        
        for report_type, file_path in files.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {report_type}: {file_path.name}")
            logger.info(f"{'='*60}")
            
            # Load file
            df = self.load_file(file_path)
            
            # Validate columns
            if not self.validate_columns(df, report_type):
                raise ValueError(f"Column validation failed for {report_type}")
            
            # Store report
            reports[report_type] = df
            
            # Log summary
            logger.info(f"✓ Validated {report_type}")
            logger.info(f"  - Rows: {len(df)}")
            logger.info(f"  - Columns: {len(df.columns)}")
            logger.info(f"  - Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return reports


# Usage
if __name__ == "__main__":
    # Initialize loader
    loader = AmazonAdsDataLoader(data_directory='./amazon_reports')
    
    # Load all reports
    reports = loader.load_all_reports()
    
    print("\n✅ All reports loaded successfully!")
    print(f"Total reports: {len(reports)}")