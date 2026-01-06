# Amazon Ads Campaign Analytics POC

**Version:** 1.0  
**Date:** January 2026  
**Status:** Proof of Concept

---

## 📋 Executive Summary

This POC demonstrates an end-to-end Amazon Sponsored Products campaign analytics platform that:
- Ingests Amazon Ads performance data into a high-speed search engine (Typesense)
- Provides real-time campaign search and analysis through an interactive dashboard
- Generates AI-powered optimization recommendations using GPT-4

---

## 🎯 What This System Does

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Data Ingestion** | Loads Amazon Ads reports (CSV/XLSX) into Typesense for sub-millisecond queries |
| **Campaign Search** | Search across all report types by campaign name with instant results |
| **Keyword Analysis** | Identify best-performing and underperforming keywords based on CTR |
| **AI Recommendations** | Generate actionable optimization insights using OpenAI GPT models |

### User Workflow

```
[Amazon Ads Reports] → [Typesense Ingestion] → [Dashboard Search] → [AI Analysis]
         ↓                      ↓                      ↓                 ↓
   CSV/XLSX files        Fast indexing         Real-time lookup    GPT recommendations
```

---

## 📁 Data Sources

### Input Files (Located in `./data/`)

| File | Description | Key Fields |
|------|-------------|------------|
| `Sponsored_Products_Budget_L30.csv` | Daily budget allocations | Campaign, Budget Amount |
| `Sponsored_Products_Campaign_L30.csv` | Campaign-level performance | Impressions, Clicks, Spend, Sales |
| `Sponsored_Products_Placement_L30.xlsx` | Performance by ad placement | Top of Search, Product Pages, Rest of Search |
| `Sponsored_Products_Search_Term_Detailed_L30.xlsx` | Search term performance | Customer Search Term, Impressions, Clicks, CTR |
| `Sponsored_Products_Targeting_Detailed_L30.xlsx` | Targeting strategy data | Match Type, Targeting, Bid |

### Data Flow

```
Raw Excel/CSV Files
        ↓
┌───────────────────────────┐
│  typesense_ingest.py      │  ← Column normalization (snake_case)
│  - Load files             │  ← Currency symbol stripping ($)
│  - Normalize columns      │  ← Stable ID generation
│  - Create collections     │  ← Type inference (string/int/float)
└───────────────────────────┘
        ↓
┌───────────────────────────┐
│     Typesense Server      │  ← Fast in-memory search
│  - 5 Collections          │  ← Sub-millisecond queries
│  - Full-text search       │  ← Filter by campaign
└───────────────────────────┘
        ↓
┌───────────────────────────┐
│   Streamlit Dashboard     │  ← Real-time visualization
│   & GPT Integration       │  ← AI recommendations
└───────────────────────────┘
```

---

## 📊 Calculations & Metrics

### Metric Formulas

All calculations are performed **before** data is sent to GPT. The AI receives pre-computed metrics only.

| Metric | Formula | Location |
|--------|---------|----------|
| **CTR** (Click-Through Rate) | `(clicks / impressions) × 100` | `campaign_recommendations.py:154` |
| **CPC** (Cost Per Click) | `spend / clicks` | `campaign_recommendations.py:155` |
| **ACOS** (Advertising Cost of Sales) | `(spend / sales) × 100` | `campaign_recommendations.py:156` |
| **ROAS** (Return on Ad Spend) | `sales / spend` | `campaign_recommendations.py:157` |
| **CVR** (Conversion Rate) | `sales / clicks` | `campaign_recommendations.py:410` |

### Keyword Classification Logic

#### Best Performing Keywords (`campaign_recommendations.py:162-166`)
```python
# Keywords with highest CTR that have actual clicks
filtered.sort_values(["ctr", "impressions"], ascending=[False, False])
```
Criteria:
- Must meet minimum impression threshold (configurable, default: 50)
- Sorted by CTR (highest first), then by impressions

#### Underperforming Keywords (`campaign_recommendations.py:167-171`)
```python
# Keywords with lowest CTR despite high impressions
filtered.sort_values(["ctr", "impressions"], ascending=[True, False])
```
Criteria:
- High impressions but low/zero CTR = wasted ad spend
- These are budget-burning keywords that need attention

### Dashboard Keyword Analysis (`typesense_campaign_dashboard.py`)

| Category | Logic |
|----------|-------|
| **Top Performing** | `clicks > 0` AND `CTR > threshold` → Sort by CTR desc, impressions desc |
| **Underperforming** | `impressions >= min` AND (`CTR ≤ threshold` OR `clicks = 0`) → Sort by impressions desc, CTR asc |

The low CTR threshold is configurable via sidebar slider (default: 1.0%).

---

## 🏗️ System Architecture

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Search Engine** | Typesense 0.25.1 | Fast full-text search, millisecond queries |
| **Dashboard** | Streamlit | Interactive Python web app |
| **AI Engine** | OpenAI GPT-4o-mini | Natural language recommendations |
| **Data Processing** | Pandas | DataFrame operations, aggregations |
| **Containerization** | Docker Compose | Multi-service orchestration |
| **Configuration** | python-dotenv | Environment variable management |

### File Structure

```
📁 n8n-work-flow/
├── 📄 typesense_campaign_dashboard.py    # Main dashboard application
├── 📄 campaign_recommendations.py         # AI recommendation engine
├── 📄 typesense_ingest.py                 # Data ingestion script
├── 📄 docker-compose.yml                  # Container orchestration
├── 📄 requirements.txt                    # Python dependencies
├── 📄 .env                                # Environment variables (create from env.example)
├── 📁 data/                               # Input: Amazon Ads reports
├── 📁 output/recommendations/             # Output: AI-generated markdown files
└── 📁 typesense-data/                     # Typesense persistent storage
```

### Key Files

| File | Purpose | Entry Points |
|------|---------|--------------|
| `typesense_campaign_dashboard.py` | Main UI application | `streamlit run typesense_campaign_dashboard.py` |
| `campaign_recommendations.py` | GPT integration & metrics | `generate_recommendations()`, `generate_recommendations_streaming()` |
| `typesense_ingest.py` | Data loading pipeline | `python typesense_ingest.py --api-key mykey` |

---

## 🔧 Setup & Configuration

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- OpenAI API key

### Environment Variables

Create a `.env` file from `env.example`:

```env
# Typesense Configuration
TYPESENSE_HOST=localhost
TYPESENSE_PORT=8108
TYPESENSE_PROTOCOL=http
TYPESENSE_API_KEY=mykey

# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-key-here
```

### Quick Start

```bash
# 1. Start Typesense server
docker-compose up typesense -d

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Ingest data into Typesense
python typesense_ingest.py --api-key mykey --drop-existing

# 4. Run the dashboard
streamlit run typesense_campaign_dashboard.py
```

### Docker Deployment

```bash
# Start both Typesense and Streamlit
docker-compose up -d

# Access dashboard at http://localhost:8501
```

---

## 📱 Dashboard Features

### Tab 1: Campaign Data
- Multi-collection search across all data sources
- Filter results by campaign name
- View raw data in expandable tables
- Results persist across tab switches

### Tab 2: Keyword Analysis
- Configurable thresholds (min impressions, CTR threshold)
- Best performing keywords (high CTR + clicks)
- Underperforming keywords (high impressions, low CTR)
- Includes match type and targeting information
- Aggregated by unique keyword (grouped stats)

### Tab 3: AI Recommendations
- Streaming GPT responses (real-time display)
- Automatic save to markdown files
- Copy-to-clipboard functionality
- Structured recommendations with:
  - Budget optimization
  - Bid strategy adjustments
  - Placement optimization
  - Keyword expansion opportunities
  - Negative keyword suggestions
  - Quick wins

---

## 📈 AI Recommendation Categories

The GPT model analyzes pre-computed metrics and provides recommendations in these areas:

| Category | Focus |
|----------|-------|
| **Budget Optimization** | Utilization rate, daily pacing, budget constraints |
| **Bid Strategy** | CPC analysis, bid efficiency, keyword-level adjustments |
| **Placement Optimization** | Top of Search vs Product Pages performance |
| **Keyword Expansion** | High-potential keywords to scale |
| **Negative Keywords** | Wasted spend elimination, irrelevant traffic prevention |
| **Performance Anomalies** | Unusual metrics, potential issues |
| **Quick Wins** | Immediate high-impact actions |

---

## 🔒 Security Notes

- API keys stored in `.env` file (not committed to Git)
- `.env` is in `.gitignore`
- Typesense runs locally (not exposed to internet)
- OpenAI API calls made server-side only

---

## 📝 Output Files

AI recommendations are saved to:
```
./output/recommendations/recommendations_{campaign_name}_{timestamp}.md
```

Example output:
```markdown
# Campaign Recommendations: My Campaign

**Generated:** 2026-01-06 15:12:55

---

**[HIGH] Budget: Increase Daily Budget**
- Current State: $50/day budget, 98% utilization
- Recommended Action: Increase budget to $75/day (+50%)
- Expected Impact: 3-5 additional conversions/week
- Implementation: Adjust in Campaign Settings > Daily Budget
...
```

---

## 🚀 Future Enhancements (Roadmap)

- [ ] Historical trend analysis
- [ ] Automated scheduling of recommendations
- [ ] Multi-campaign comparison
- [ ] Export to Amazon Ads bulk upload format
- [ ] Integration with n8n workflows
- [ ] Custom GPT fine-tuning for Amazon Ads domain

---

## 📞 Support

For questions about this POC, refer to:
- Terminal logs for debugging (enabled by default)
- Typesense admin API: `http://localhost:8108/health`
- OpenAI API dashboard for usage monitoring

---

*This POC demonstrates the viability of AI-powered Amazon Ads optimization. Production deployment would require additional security hardening, scalability testing, and integration with Amazon Ads API for real-time data.*
