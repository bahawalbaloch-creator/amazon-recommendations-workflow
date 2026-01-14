"""merge bid to targeting term """

import pandas as pd
import numpy as np

def merge_bid_to_keywords(bid_file, bedsheets_file, output_file):
    """
    Merge bid data from bid_data_only.csv to Bedsheets search term file.
    
    Matches on: portfolio, campaign, ad_group, targeting
    Bid column names supported (checked in order): Bid, bid
    Default bid column names supported (checked in order):
    Ad Group Default Bid (Informational only), ad_group_default_bid_, ad_group_default_bid
    Campaign column aliases: Campaign Name (Informational only), campaign
    Ad group column aliases: Ad Group Name (Informational only), ad_group_name_, ad_group
    Targeting aliases: keyword_text, resolved_product_targeting_expression_
    """
    def _get_first_existing(df, candidates, required=True):
        """Return the first column in candidates that exists in df, else raise/None."""
        for col in candidates:
            if col in df.columns:
                return col
        if required:
            raise KeyError(f"None of the columns {candidates} found in DataFrame")
        return None

    print("Reading bid data file...")
    bid_df = pd.read_csv(bid_file)
    
    print("Reading Bedsheets file...")
    bedsheets_df = pd.read_csv(bedsheets_file)
    
    print(f"Bid data shape: {bid_df.shape}")
    print(f"Bedsheets data shape: {bedsheets_df.shape}")
    
    # Resolve column names that may vary between exports
    bid_col = _get_first_existing(bid_df, ['Bid', 'bid'])
    default_bid_col = _get_first_existing(
        bid_df,
        ['Ad Group Default Bid (Informational only)', 'ad_group_default_bid_', 'ad_group_default_bid'],
        required=False,
    )
    bid_portfolio_col = _get_first_existing(
        bid_df,
        ['Portfolio Name (Informational only)', 'Portfolio Name', 'portfolio_name', 'portfolio']
    )
    bid_campaign_col = _get_first_existing(
        bid_df,
        ['Campaign Name (Informational only)', 'campaign']
    )
    bid_ad_group_col = _get_first_existing(
        bid_df,
        ['Ad Group Name (Informational only)', 'ad_group_name_', 'ad_group']
    )
    targeting_col = _get_first_existing(
        bid_df,
        ['keyword_text', 'resolved_product_targeting_expression_']
    )
    
    # Create a matching key in bid_df
    bid_df['match_key'] = (
        bid_df[bid_portfolio_col].astype(str) + '|' +
        bid_df[bid_campaign_col].astype(str) + '|' +
        bid_df[bid_ad_group_col].astype(str) + '|' +
        bid_df[targeting_col].astype(str)
    )
    
    # Create a matching key in bedsheets_df
    bedsheets_df['match_key'] = (
        bedsheets_df['portfolio_name'].astype(str) + '|' +
        bedsheets_df['campaign'].astype(str) + '|' +
        bedsheets_df['ad_group'].astype(str) + '|' +
        bedsheets_df['targeting'].astype(str)
    )
    
    # Create a lookup dictionary for bids (with full match key)
    # Use Bid if not empty, otherwise use Ad Group Default Bid
    bid_lookup = {}
    for idx, row in bid_df.iterrows():
        key = row['match_key']
        bid_value = row[bid_col]
        default_bid = row[default_bid_col] if default_bid_col else None
        
        # Use bid if it's not empty/NaN, otherwise use default bid
        if pd.notna(bid_value) and bid_value != '':
            bid_lookup[key] = bid_value
        elif default_bid_col and pd.notna(default_bid) and default_bid != '':
            bid_lookup[key] = default_bid
        else:
            bid_lookup[key] = None
    
    # Also create a fallback lookup for ad_group default bids (without targeting)
    # This is used when exact targeting match is not found
    ad_group_default_lookup = {}
    for idx, row in bid_df.iterrows():
        portfolio = str(row[bid_portfolio_col])
        campaign = str(row[bid_campaign_col])
        ad_group = str(row[bid_ad_group_col])
        default_bid = row[default_bid_col] if default_bid_col else None
        
        fallback_key = f"{portfolio}|{campaign}|{ad_group}"
        if default_bid_col and pd.notna(default_bid) and default_bid != '':
            # Keep the first non-null default bid for each ad_group
            if fallback_key not in ad_group_default_lookup:
                ad_group_default_lookup[fallback_key] = default_bid
    
    print(f"Created bid lookup with {len(bid_lookup)} entries")
    print(f"Created ad_group fallback lookup with {len(ad_group_default_lookup)} entries")
    
    # Map bids to bedsheets_df - first try exact match, then fallback to ad_group default
    bedsheets_df['bid'] = bedsheets_df['match_key'].map(bid_lookup)
    
    # For rows without exact match, try ad_group fallback
    missing_mask = bedsheets_df['bid'].isna()
    if missing_mask.any():
        bedsheets_df['fallback_key'] = (
            bedsheets_df['portfolio_name'].astype(str) + '|' +
            bedsheets_df['campaign'].astype(str) + '|' +
            bedsheets_df['ad_group'].astype(str)
        )
        bedsheets_df.loc[missing_mask, 'bid'] = bedsheets_df.loc[missing_mask, 'fallback_key'].map(ad_group_default_lookup)
        bedsheets_df = bedsheets_df.drop(columns=['fallback_key'])
    
    # Count how many matches we found
    matched_count = bedsheets_df['bid'].notna().sum()
    total_count = len(bedsheets_df)
    print(f"Matched bids: {matched_count} out of {total_count} rows ({matched_count/total_count*100:.2f}%)")
    
    # Remove the temporary match_key column
    bedsheets_df = bedsheets_df.drop(columns=['match_key'])
    
    # Reorder columns to put bid after targeting
    cols = bedsheets_df.columns.tolist()
    if 'bid' in cols:
        cols.remove('bid')
        # Find targeting column index
        targeting_idx = cols.index('targeting') if 'targeting' in cols else len(cols)
        cols.insert(targeting_idx + 1, 'bid')
        bedsheets_df = bedsheets_df[cols]
    
    print(f"Saving merged data to {output_file}...")
    bedsheets_df.to_csv(output_file, index=False)
    print("Done!")
    
    return bedsheets_df

if __name__ == "__main__":
    bid_file = "bid_data_only.csv"
    bedsheets_file = "portfolio_split_output/Bedsheets_Sponsored_Products_Search_Term_Detailed_L30.csv"
    output_file = "portfolio_split_output/Bedsheets_Sponsored_Products_Search_Term_Detailed_L30_with_bids.csv"
    
    merge_bid_to_keywords(bid_file, bedsheets_file, output_file)

