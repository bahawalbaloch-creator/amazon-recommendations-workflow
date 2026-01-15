import pandas as pd
from pathlib import Path

# -------- Config --------
excel_file = "bulk-a3aqp8tdyvycgl-20251114-20251114-1767357245518.xlsx"
sheet_name = "Sponsored Products Campaigns 1"

campaign_col = "Keyword ID"
keyword_col = "Ad Group Default Bid (Informational only)"

output_csv = "keyword_id_to_bid_mapping.csv"
# ------------------------

# Read Excel
df = pd.read_excel(excel_file, sheet_name=sheet_name)
df = df[df['Entity'] == 'Keyword']
# Keep only required columns
campaign_to_keyword = (
    df[[campaign_col, keyword_col]]
    .dropna(subset=[campaign_col, keyword_col])   # remove empty values
    .drop_duplicates()                             # avoid duplicate pairs
    .rename(columns={
        campaign_col: "keyword_id",
        keyword_col: "keyword_bid"
    })
)

# Save to CSV
campaign_to_keyword.to_csv(output_csv, index=False)

print(f"CSV created successfully: {Path(output_csv).resolve()}")
print(f"Total unique campaign-keyword pairs: {len(campaign_to_keyword)}")
