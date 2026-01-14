import pandas as pd

# --- CONFIG ---
input_file = 'bulk-a3aqp8tdyvycgl-20251114-20251114-1767357245518.xlsx'  # <-- change this to your XLSX file path
sheet_name = "Sponsored Products Campaigns 1"
output_file = 'bid_data.csv'  # <-- output CSV file path

# Columns to extract
columns_to_extract = [
    "Entity",
    "Campaign Name (Informational only)",
    "Ad Group Name (Informational only)",
    "Portfolio Name (Informational only)",
    "Ad Group Default Bid (Informational only)",
    "Bid",
    "Keyword Text",
    "Resolved Product Targeting Expression (Informational only)",
    "Impressions",
    "Clicks",
    "Click-through Rate",
    "Spend",
    "Sales",
    "Orders",
    "Units",
    "Conversion Rate",
    "ACOS",
    "CPC",
    "ROAS"
]

# --- LOAD EXCEL ---
try:
    df = pd.read_excel(input_file, sheet_name=sheet_name, engine='openpyxl')
except Exception as e:
    print(f"Error reading Excel file: {e}")
    exit()

# --- FILTER COLUMNS ---
missing_cols = [col for col in columns_to_extract if col not in df.columns]
if missing_cols:
    print(f"Warning: The following columns are missing in the sheet: {missing_cols}")

df_filtered = df[[col for col in columns_to_extract if col in df.columns]]

# --- SAVE TO CSV ---
df_filtered.to_csv(output_file, index=False)
print(f"Data extracted successfully to {output_file}")
