import pandas as pd
from pathlib import Path
import re

# Read the Excel file with the specific sheet
excel_file = 'bulk-a3aqp8tdyvycgl-20251114-20251114-1767357245518.xlsx'
sheet_name = 'SP Search Term Report'

print(f"Reading {excel_file} from sheet '{sheet_name}'...")

# Read the specific sheet
df = pd.read_excel(excel_file, sheet_name=sheet_name, engine='openpyxl')

print(f"Loaded {len(df)} rows")
print(f"Total columns: {len(df.columns)}")

# Define column pairs to extract (ID column, Name column)
# Format: (id_pattern, name_pattern, output_id_name, output_name_name)
column_pairs = [
    ('Campaign Name (Informational only)', 'Ad Group Name', 'Campaign Name', 'Ad Group Name'),
    ('Campaign ID', 'Ad Group ID', 'Campaign ID', 'Ad Group ID'),
    # ('Campaign ID', 'Ad Group ID', 'Campaign ID', 'Ad Group ID'),
    # ('Ad Group ID', 'Ad Group Name (Informational only)', 'Ad Group ID', 'Ad Group Name (Informational only)'),
    # ('Campaign ID', 'Campaign Name (Informational only)', 'Campaign ID', 'Campaign Name (Informational only)'),
]

def find_column(df, pattern):
    """Find a column that contains the pattern (exact match preferred)"""
    exact_match = None
    partial_match = None
    
    for col in df.columns:
        col_str = str(col)
        if col_str == pattern:
            exact_match = col
            break
        elif pattern in col_str:
            partial_match = col
    
    return exact_match if exact_match else partial_match

# Process each column pair
output_files = []

for id_pattern, name_pattern, output_id_name, output_name_name in column_pairs:
    print(f"\n{'='*60}")
    print(f"Processing pair: {output_id_name} -> {output_name_name}")
    print(f"{'='*60}")
    
    # Find ID column
    id_col = find_column(df, id_pattern)
    
    if id_col is None:
        print(f"  ⚠ Skipping: Could not find ID column matching '{id_pattern}'")
        continue
    
    # Find Name column
    name_col = None
    if name_pattern:
        name_col = find_column(df, name_pattern)
        if name_col is None:
            print(f"  ⚠ Skipping: Could not find Name column matching '{name_pattern}'")
            continue
    else:
        # If no name pattern, try to infer from ID column name
        print(f"  ⚠ No name pattern provided for {id_pattern}, skipping")
        continue
    
    print(f"  Found ID column: '{id_col}'")
    print(f"  Found Name column: '{name_col}'")
    
    # Extract unique pairs
    # Keep case-sensitive names
    unique_df = df[[id_col, name_col]].copy()
    
    # Remove rows where ID is null
    unique_df = unique_df.dropna(subset=[id_col])
    
    # Remove duplicates while preserving case sensitivity (keep first occurrence)
    initial_count = len(unique_df)
    unique_df = unique_df.drop_duplicates(subset=[id_col], keep='first')
    final_count = len(unique_df)
    
    print(f"  Initial rows: {initial_count:,}")
    print(f"  Unique IDs: {final_count:,}")
    print(f"  Duplicates removed: {initial_count - final_count:,}")
    
    # Rename columns for output
    unique_df.columns = [output_id_name, output_name_name]
    
    # Generate output filename
    safe_id_name = re.sub(r'[^\w\s-]', '', output_id_name).replace(' ', '_').lower()
    output_file = f'{safe_id_name}_to_name_map.csv'
    
    # Save to CSV
    unique_df.to_csv(output_file, index=False, encoding='utf-8')
    output_files.append(output_file)
    
    print(f"  ✓ Saved mapping to {output_file}")
    print(f"  First 5 rows:")
    print(unique_df.head(5).to_string(index=False))
    print()

print(f"\n{'='*60}")
print(f"Summary: Created {len(output_files)} mapping files")
print(f"{'='*60}")
for f in output_files:
    print(f"  ✓ {f}")
