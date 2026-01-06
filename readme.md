# Amazon Ads Optimization Pipeline

Automated pipeline for analyzing Amazon Sponsored Products reports and generating AI-powered optimization recommendations.

## Quick Start

### 1. Installation

**Linux/Mac:**
```bash
chmod +x setup.sh
./setup.sh
```

**Windows:**
```cmd
setup.bat
```

**Manual Setup:**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p amazon_reports output
```

### 2. Configuration

Edit `.env` file and add your OpenAI API key:
```
OPENAI_API_KEY=sk-your-key-here
```

### 3. Prepare Data

Download these reports from Amazon Ads Console (Last 30 Days, Daily frequency):

1. **Search Term Report** (Detailed) → `Sponsored_Products_Search_Term_Detailed_L30.xlsx`
2. **Targeting Report** (Detailed) → `Sponsored_Products_Targeting_Detailed_L30.xlsx`
3. **Campaign Report** (Summary) → `Sponsored_Products_Campaign_L30.xlsx`
4. **Placement Report** (Summary) → `Sponsored_Products_Placement_L30.xlsx`
5. **Budget Report** (Summary) → `Sponsored_Products_Budget_L30.xlsx`

Place all files in `./amazon_reports/` directory.

### 4. Run Pipeline

```bash
# Basic usage (uses .env configuration)
python run_pipeline.py

# With command line arguments
python run_pipeline.py --data-dir ./my_reports --api-key sk-xxx

# Using config file
python run_pipeline.py --config config.json

# Skip intermediate files
python run_pipeline.py --no-intermediates
```

## Output Files

The pipeline generates three types of reports in `./output/`:

### 📊 Excel Report
`amazon_ads_recommendations_YYYYMMDD_HHMMSS.xlsx`

Contains 5 worksheets:
- **Campaign Adjustments**: Bid multipliers, budget changes, projections
- **Keywords - Add Exact**: New exact-match keywords to add
- **Keywords - Negative**: Keywords to exclude
- **Targeting Adjustments**: Targets to pause or bid increases
- **Summary**: Overview of all recommendations

### 📄 JSON Report
`recommendations_YYYYMMDD_HHMMSS.json`

Raw JSON data for API integration or custom processing.

### 📝 Summary Report
`summary_YYYYMMDD_HHMMSS.txt`

Quick text overview of data and recommendations.

## Project Structure

```
amazon-ads-optimizer/
├── run_pipeline.py          # Main pipeline orchestrator
├── data_loader.py           # Load and validate reports
├── data_converter.py        # Convert and standardize data
├── data_merger.py           # Merge and aggregate data
├── llm_optimizer.py         # GPT-4o API integration
├── report_generator.py      # Generate output reports
├── requirements.txt         # Python dependencies
├── config.json             # Configuration file (optional)
├── .env                    # Environment variables
├── amazon_reports/         # Input directory
│   └── [Amazon Ads reports here]
└── output/                 # Output directory
    ├── logs/               # Pipeline logs
    ├── csv_output/         # Intermediate CSVs
    ├── aggregated_data/    # Aggregated datasets
    └── [Generated reports]
```

## Command Line Options

```
--data-dir PATH        Directory with Amazon Ads reports (default: ./amazon_reports)
--api-key KEY          OpenAI API key (or use OPENAI_API_KEY env var)
--output-dir PATH      Output directory (default: ./output)
--config FILE          JSON configuration file
--no-intermediates     Skip saving intermediate CSV files
```

## Configuration File

Create `config.json`:

```json
{
  "data_directory": "./amazon_reports",
  "output_directory": "./output",
  "api_key": "your-openai-api-key",
  "save_intermediates": true,
  "model": "gpt-4o",
  "temperature": 0.3,
  "max_tokens": 4000
}
```

## Troubleshooting

### Missing Files Error
Ensure all 5 required reports are in the data directory with correct naming patterns.

### API Key Error
- Verify your OpenAI API key is valid
- Check you have sufficient API credits
- Ensure key has access to GPT-4o

### Column Validation Error
Amazon may change column names. Check the log file in `output/logs/` for details.

### Out of Memory
For large datasets (>100K rows), increase system RAM or filter data to last 30 days only.

## Support

For issues or questions:
1. Check log files in `output/logs/`
2. Review error messages in console
3. Verify all prerequisites are met

## License