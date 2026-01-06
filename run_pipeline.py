"""
Amazon Ads Optimization Pipeline
Complete end-to-end workflow for analyzing Amazon Ads reports and generating recommendations

Usage:
    python run_pipeline.py --data-dir ./amazon_reports --api-key YOUR_KEY
    python run_pipeline.py --config config.json
    python run_pipeline.py  # Uses defaults from .env file
"""

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import pandas as pd

# Import custom modules
from amazon_data_loader import AmazonAdsDataLoader
from xlxs_to_csv_convertor import DataConverter
from data_merger import DataMerger
from llm_call import LLMOptimizer
from report_generator import ReportGenerator
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()


# Configure logging
def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging configuration"""
    log_dir = output_dir / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'pipeline_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


class AmazonAdsPipeline:
    """Main pipeline orchestrator for Amazon Ads optimization"""
    
    def __init__(
        self,
        data_directory: str,
        api_key: str,
        output_directory: str = './output',
        save_intermediates: bool = True,
        campaign_filter: str = None
    ):
        self.data_dir = Path(data_directory)
        self.api_key = api_key
        self.output_dir = Path(output_directory)
        self.save_intermediates = save_intermediates
        self.campaign_filter = campaign_filter
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / 'csv_output').mkdir(exist_ok=True)
        (self.output_dir / 'aggregated_data').mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = setup_logging(self.output_dir)
        
        # Cache file for processed reports
        self.cache_file = self.output_dir / 'processed_reports_cache.pkl'
        
        # Initialize components
        self.loader = None
        self.converter = None
        self.merger = None
        self.optimizer = None
        self.generator = None
        
        # Data storage
        self.reports = {}
        self.processed_reports = {}
        self.master_data = {}
        self.recommendations = {}
        
        self.logger.info("Pipeline initialized")
        self.logger.info(f"Data directory: {self.data_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def print_banner(self, step: str, total: int, current: int):
        """Print formatted step banner"""
        banner = "=" * 80
        print(f"\n{banner}")
        print(f"[STEP {current}/{total}] {step}")
        print(banner)
        self.logger.info(f"Starting Step {current}/{total}: {step}")
    
    def get_source_files_mtime(self) -> float:
        """Get the maximum modification time of source files"""
        max_mtime = 0
        for file_path in self.data_dir.glob('*.xlsx'):
            mtime = file_path.stat().st_mtime
            max_mtime = max(max_mtime, mtime)
        for file_path in self.data_dir.glob('*.csv'):
            mtime = file_path.stat().st_mtime
            max_mtime = max(max_mtime, mtime)
        return max_mtime
    
    def is_cache_valid(self) -> bool:
        """Check if cache exists and is newer than source files"""
        if not self.cache_file.exists():
            return False
        cache_mtime = self.cache_file.stat().st_mtime
        source_mtime = self.get_source_files_mtime()
        return cache_mtime > source_mtime
    
    def save_processed_cache(self):
        """Save processed reports to cache"""
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.processed_reports, f)
        self.logger.info(f"Saved processed reports cache: {self.cache_file}")
    
    def load_processed_cache(self):
        """Load processed reports from cache"""
        with open(self.cache_file, 'rb') as f:
            self.processed_reports = pickle.load(f)
        self.logger.info(f"Loaded processed reports from cache: {self.cache_file}")
    
    def _filter_reports_by_campaign(self, reports: Dict[str, pd.DataFrame], campaign_name: str) -> Dict[str, pd.DataFrame]:
        """Filter all reports to only include data for the specified campaign"""
        filtered_reports = {}
        
        for report_type, df in reports.items():
            if 'campaign' in df.columns:
                # Case-insensitive match
                filtered_df = df[df['campaign'].str.lower() == campaign_name.lower()].copy()
                if len(filtered_df) == 0:
                    self.logger.warning(f"No data found for campaign '{campaign_name}' in {report_type}. Available campaigns: {df['campaign'].unique()[:10]}")
                else:
                    self.logger.info(f"Filtered {report_type} to {len(filtered_df)} rows for campaign '{campaign_name}'")
                filtered_reports[report_type] = filtered_df
            else:
                # For reports without campaign column, include as-is (like budgets if not per campaign)
                filtered_reports[report_type] = df.copy()
        
        return filtered_reports
    
    def step1_load_data(self) -> bool:
        """Step 1: Load Amazon Ads reports"""
        self.print_banner("LOADING AMAZON ADS REPORTS", 5, 1)
        
        try:
            self.loader = AmazonAdsDataLoader(data_directory=str(self.data_dir))
            self.reports = self.loader.load_all_reports()
            
            # Filter by campaign if specified
            if self.campaign_filter:
                self.reports = self._filter_reports_by_campaign(self.reports, self.campaign_filter)
            
            # Print summary
            print("\n✅ Data Loading Summary:")
            for report_type, df in self.reports.items():
                print(f"  • {report_type.ljust(20)}: {len(df):,} rows, {len(df.columns)} columns")
                self.logger.info(f"Loaded {report_type}: {len(df)} rows")
            
            return True
            
        except FileNotFoundError as e:
            self.logger.error(f"Missing files: {str(e)}")
            print(f"\n❌ Error: {str(e)}")
            print("\nPlease ensure all required files are in the data directory:")
            print("  1. Sponsored_Products_Search_Term_Detailed_L30.xlsx")
            print("  2. Sponsored_Products_Targeting_Detailed_L30.xlsx")
            print("  3. Sponsored_Products_Campaign_L30.xlsx")
            print("  4. Sponsored_Products_Placement_L30.xlsx")
            print("  5. Sponsored_Products_Budget_L30.xlsx")
            return False
            
        except Exception as e:
            self.logger.error(f"Data loading failed: {str(e)}", exc_info=True)
            print(f"\n❌ Error loading data: {str(e)}")
            return False
    
    def step2_process_data(self) -> bool:
        """Step 2: Convert and standardize data"""
        self.print_banner("CONVERTING & STANDARDIZING DATA", 5, 2)
        
        try:
            # Check if we can use cached processed data
            if self.is_cache_valid():
                print("📋 Using cached processed data...")
                self.load_processed_cache()
                print("✅ Loaded from cache - skipping processing step")
                return True
            
            # Process data if no valid cache
            csv_output_dir = str(self.output_dir / 'csv_output')
            self.converter = DataConverter(output_directory=csv_output_dir)
            self.processed_reports = self.converter.process_all_reports(self.reports)
            
            # Save to cache for future runs
            self.save_processed_cache()
            
            # Save CSVs if requested
            if self.save_intermediates:
                csv_files = self.converter.convert_xlsx_to_csv(self.processed_reports)
                print("\n✅ CSV Files Generated:")
                for report_type, file_path in csv_files.items():
                    print(f"  • {file_path.name}")
            
            # Print metrics summary
            print("\n✅ Calculated Metrics:")
            sample_df = next(iter(self.processed_reports.values()))
            metric_cols = [col for col in sample_df.columns if col in ['ctr', 'cpc', 'acos', 'roas']]
            for col in metric_cols:
                print(f"  • {col.upper()}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Data processing failed: {str(e)}", exc_info=True)
            print(f"\n❌ Error processing data: {str(e)}")
            return False
    
    def step3_merge_data(self) -> bool:
        """Step 3: Merge and aggregate data"""
        self.print_banner("MERGING & AGGREGATING DATA", 5, 3)
        
        try:
            self.merger = DataMerger(self.processed_reports)
            self.master_data = self.merger.create_master_dataset()
            
            # Save aggregated data if requested
            if self.save_intermediates:
                agg_dir = self.output_dir / 'aggregated_data'
                for name, df in self.master_data.items():
                    file_path = agg_dir / f"{name}.csv"
                    df.to_csv(file_path, index=False)
                    self.logger.info(f"Saved aggregated data: {file_path}")
            
            # Print summary
            print("\n✅ Aggregated Datasets Created:")
            for name, df in self.master_data.items():
                print(f"  • {name.ljust(25)}: {len(df):,} rows")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Data merging failed: {str(e)}", exc_info=True)
            print(f"\n❌ Error merging data: {str(e)}")
            return False
    
    def step4_get_recommendations(self) -> bool:
        """Step 4: Call GPT-4o for recommendations"""
        self.print_banner("GENERATING AI RECOMMENDATIONS", 5, 4)
        
        try:
            self.optimizer = LLMOptimizer(api_key=self.api_key, campaign_filter=self.campaign_filter)
            
            # Prepare data
            print("\n⏳ Preparing data for AI analysis...")
            data_payload = self.optimizer.prepare_data_for_llm(self.processed_reports)
            payload_size = len(json.dumps(data_payload))
            print(f"  • Payload size: {payload_size:,} characters")
            self.logger.info(f"Data payload prepared: {payload_size} characters")
            
            # Save data payload for inspection
            payload_file = self.output_dir / 'llm_data_payload.json'
            with open(payload_file, 'w', encoding='utf-8') as f:
                json.dump(data_payload, f, indent=2, ensure_ascii=False)
            print(f"  • Data payload saved to: {payload_file}")
            

            print("\n⏳ Calling GPT-4o API (this may take 30-60 seconds)...")
            campaign = self.campaign_filter

            winners = [
                row for row in data_payload['search_term_winners']
                if row['campaign'] == campaign
            ]

            losers = [
                row for row in data_payload['search_term_losers']
                if row['campaign'] == campaign
            ]

            single_campaign_data = winners + losers
            print(f"  • Filtered to {len(single_campaign_data)} records for campaign '{single_campaign_data}'")

            self.recommendations = self.optimizer.call_gpt4o(single_campaign_data)
            
            # Validate
            if not self.optimizer.validate_recommendations(self.recommendations):
                raise ValueError("Recommendations validation failed")
            
            # Print summary
            print("\n✅ AI Recommendations Generated:")
            print(f"  • Campaign Adjustments: {len(self.recommendations['campaign_adjustments'])}")
            print(f"  • Keywords to Add: {len(self.recommendations['keyword_recommendations']['add_exact'])}")
            print(f"  • Negative Keywords: {len(self.recommendations['keyword_recommendations']['negative'])}")
            print(f"  • Targeting Changes: {len(self.recommendations['targeting_recommendations'])}")
            if 'Detailed reasoning' in self.recommendations:
                print(f"  • Detailed reasoning: {self.recommendations['Detailed reasoning'][:100]}...")
            
            return True
            
        except Exception as e:
            self.logger.error(f"AI recommendations failed: {str(e)}", exc_info=True)
            print(f"\n❌ Error getting recommendations: {str(e)}")
            
            if "api_key" in str(e).lower():
                print("\n💡 Tip: Make sure your OpenAI API key is valid and has sufficient credits")
            
            return False
    
    def step5_generate_reports(self) -> bool:
        """Step 5: Generate output reports"""
        self.print_banner("GENERATING OUTPUT REPORTS", 5, 5)
        
        try:
            self.generator = ReportGenerator(
                recommendations=self.recommendations,
                output_directory=str(self.output_dir)
            )
            
            # Generate reports
            print("\n⏳ Generating Excel report...")
            excel_file = self.generator.generate_excel_report()
            
            print("⏳ Generating JSON report...")
            json_file = self.generator.generate_json_report()
            
            print("⏳ Generating summary report...")
            summary_file = self.generator.generate_summary_report(self.processed_reports)
            
            # Print results
            print("\n✅ Reports Generated Successfully:")
            print(f"\n  📊 EXCEL REPORT:")
            print(f"     {excel_file}")
            print(f"     Open in Excel/Google Sheets for detailed recommendations")
            
            print(f"\n  📄 JSON REPORT:")
            print(f"     {json_file}")
            print(f"     Raw data for API integration or custom processing")
            
            print(f"\n  📝 SUMMARY REPORT:")
            print(f"     {summary_file}")
            print(f"     Quick overview of all recommendations")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}", exc_info=True)
            print(f"\n❌ Error generating reports: {str(e)}")
            return False
    
    def run(self) -> bool:
        """Execute complete pipeline"""
        start_time = datetime.now()
        
        print("\n" + "=" * 80)
        print(" " * 20 + "AMAZON ADS OPTIMIZATION PIPELINE")
        print("=" * 80)
        print(f"\nStarted: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Execute steps
        steps = [
            self.step1_load_data,
            self.step2_process_data,
            self.step3_merge_data,
            self.step4_get_recommendations,
            self.step5_generate_reports
        ]
        
        for i, step_func in enumerate(steps, 1):
            success = step_func()
            if not success:
                print("\n" + "=" * 80)
                print(f"❌ PIPELINE FAILED AT STEP {i}")
                print("=" * 80)
                self.logger.error(f"Pipeline failed at step {i}")
                return False
        
        # Success!
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 80)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nTotal execution time: {duration:.1f} seconds")
        print(f"Output directory: {self.output_dir.absolute()}")
        
        self.logger.info(f"Pipeline completed successfully in {duration:.1f} seconds")
        
        # Print next steps
        print("\n📋 NEXT STEPS:")
        print("  1. Open the Excel report to review recommendations")
        print("  2. Implement changes in Amazon Ads console")
        print("  3. Monitor performance over the next 7-14 days")
        print("  4. Re-run pipeline to assess impact\n")
        
        return True


def load_config(config_file: str) -> Dict:
    """Load configuration from JSON file"""
    with open(config_file, 'r') as f:
        return json.load(f)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Amazon Ads Optimization Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --data-dir ./reports --api-key sk-xxx
  python run_pipeline.py --config config.json
  python run_pipeline.py  # Uses environment variables
        """
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data',
        help='Directory containing Amazon Ads report files (default: ./data)'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        help='OpenAI API key (or set OPENAI_API_KEY env variable)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output',
        help='Directory for output files (default: ./output)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to JSON configuration file'
    )
    
    parser.add_argument(
        '--no-intermediates',
        action='store_true',
        help='Skip saving intermediate CSV and aggregated files'
    )
    
    parser.add_argument(
        '--campaign',
        type=str,
        help='Filter pipeline to run only on specified campaign name'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
        data_dir = config.get('data_directory', args.data_dir)
        api_key = config.get('api_key', args.api_key)
        output_dir = config.get('output_directory', args.output_dir)
        campaign_filter = config.get('campaign_filter', args.campaign)
    else:
        data_dir = args.data_dir
        api_key = args.api_key or os.getenv('OPENAI_API_KEY')
        output_dir = args.output_dir
        campaign_filter = args.campaign
    
    # Validate API key
    if not api_key:
        print("❌ Error: OpenAI API key not provided")
        print("\nProvide API key using one of these methods:")
        print("  1. Command line: --api-key YOUR_KEY")
        print("  2. Environment variable: export OPENAI_API_KEY=YOUR_KEY")
        print("  3. Config file: --config config.json")
        print("  4. .env file: Add OPENAI_API_KEY=YOUR_KEY")
        sys.exit(1)
    
    # Validate data directory
    if not Path(data_dir).exists():
        print(f"❌ Error: Data directory not found: {data_dir}")
        print(f"\nCreate the directory and add your Amazon Ads report files:")
        print(f"  mkdir {data_dir}")
        sys.exit(1)
    
    # Run pipeline
    try:
        pipeline = AmazonAdsPipeline(
            data_directory=data_dir,
            api_key=api_key,
            output_directory=output_dir,
            save_intermediates=not args.no_intermediates,
            campaign_filter=campaign_filter
        )
        
        success = pipeline.run()
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        logging.exception("Unexpected error in pipeline")
        sys.exit(1)


if __name__ == "__main__":
    main()