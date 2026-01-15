import sqlite3
import pandas as pd
import hashlib
from datetime import datetime
from pathlib import Path

class SPDataWarehouseETL:
    def __init__(self, db_path='sp_data_warehouse.db'):
        """Initialize the ETL process with database connection"""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        
        # Dimension key caches to avoid lookups
        self.dim_cache = {
            'date': {},
            'campaign': {},
            'ad_group': {},
            'targeting': {},
            'search_term': {},
            'placement': {},
            'retailer': {},
            'country': {},
            'currency': {}
        }
    
    def create_schema(self):
        """Create star schema tables"""
        print("Creating star schema...")
        
        # Dimension Tables
        self.cursor.executescript("""
            -- Dim_Date
            CREATE TABLE IF NOT EXISTS Dim_Date (
                date_key INTEGER PRIMARY KEY,
                full_date DATE UNIQUE NOT NULL,
                year INTEGER,
                quarter INTEGER,
                month INTEGER,
                month_name TEXT,
                week INTEGER,
                day_of_week INTEGER,
                day_of_month INTEGER,
                is_weekend BOOLEAN
            );
            
            -- Dim_Campaign
            CREATE TABLE IF NOT EXISTS Dim_Campaign (
                campaign_key INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_name TEXT UNIQUE NOT NULL,
                portfolio_name TEXT,
                program_type TEXT,
                targeting_type TEXT,
                bidding_strategy TEXT,
                status TEXT,
                start_date DATE,
                end_date DATE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Dim_Ad_Group
            CREATE TABLE IF NOT EXISTS Dim_Ad_Group (
                ad_group_key INTEGER PRIMARY KEY AUTOINCREMENT,
                ad_group_name TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Dim_Targeting
            CREATE TABLE IF NOT EXISTS Dim_Targeting (
                targeting_key INTEGER PRIMARY KEY AUTOINCREMENT,
                targeting_value TEXT NOT NULL,
                match_type TEXT,
                unique_hash TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Dim_Search_Term
            CREATE TABLE IF NOT EXISTS Dim_Search_Term (
                search_term_key INTEGER PRIMARY KEY AUTOINCREMENT,
                search_term_value TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Dim_Placement
            CREATE TABLE IF NOT EXISTS Dim_Placement (
                placement_key INTEGER PRIMARY KEY AUTOINCREMENT,
                placement_name TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Dim_Retailer
            CREATE TABLE IF NOT EXISTS Dim_Retailer (
                retailer_key INTEGER PRIMARY KEY AUTOINCREMENT,
                retailer_name TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Dim_Country
            CREATE TABLE IF NOT EXISTS Dim_Country (
                country_key INTEGER PRIMARY KEY AUTOINCREMENT,
                country_code TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Dim_Currency
            CREATE TABLE IF NOT EXISTS Dim_Currency (
                currency_key INTEGER PRIMARY KEY AUTOINCREMENT,
                currency_code TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Fact_SP_Performance
            CREATE TABLE IF NOT EXISTS Fact_SP_Performance (
                fact_key INTEGER PRIMARY KEY AUTOINCREMENT,
                date_key INTEGER,
                campaign_key INTEGER,
                ad_group_key INTEGER,
                targeting_key INTEGER,
                search_term_key INTEGER,
                placement_key INTEGER,
                retailer_key INTEGER,
                country_key INTEGER,
                currency_key INTEGER,
                impressions INTEGER,
                clicks INTEGER,
                click_thru_rate_ctr REAL,
                cost_per_click_cpc REAL,
                spend REAL,
                sales REAL,
                total_acos REAL,
                total_roas REAL,
                top_of_search_impression_share REAL,
                seven_day_total_orders INTEGER,
                seven_day_total_units INTEGER,
                seven_day_conversion_rate REAL,
                seven_day_advertised_sku_units INTEGER,
                seven_day_other_sku_units INTEGER,
                seven_day_advertised_sku_sales REAL,
                seven_day_other_sku_sales REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (date_key) REFERENCES Dim_Date(date_key),
                FOREIGN KEY (campaign_key) REFERENCES Dim_Campaign(campaign_key),
                FOREIGN KEY (ad_group_key) REFERENCES Dim_Ad_Group(ad_group_key),
                FOREIGN KEY (targeting_key) REFERENCES Dim_Targeting(targeting_key),
                FOREIGN KEY (search_term_key) REFERENCES Dim_Search_Term(search_term_key),
                FOREIGN KEY (placement_key) REFERENCES Dim_Placement(placement_key),
                FOREIGN KEY (retailer_key) REFERENCES Dim_Retailer(retailer_key),
                FOREIGN KEY (country_key) REFERENCES Dim_Country(country_key),
                FOREIGN KEY (currency_key) REFERENCES Dim_Currency(currency_key)
            );
            
            -- Fact_SP_Budget
            CREATE TABLE IF NOT EXISTS Fact_SP_Budget (
                budget_fact_key INTEGER PRIMARY KEY AUTOINCREMENT,
                start_date_key INTEGER,
                end_date_key INTEGER,
                campaign_key INTEGER,
                country_key INTEGER,
                currency_key INTEGER,
                budget REAL,
                recommended_budget REAL,
                average_time_in_budget REAL,
                impressions INTEGER,
                last_year_impressions INTEGER,
                estimated_missed_impressions_min INTEGER,
                estimated_missed_impressions_max INTEGER,
                clicks INTEGER,
                last_year_clicks INTEGER,
                estimated_missed_clicks_min INTEGER,
                estimated_missed_clicks_max INTEGER,
                click_thru_rate_ctr REAL,
                spend REAL,
                last_year_spend REAL,
                cost_per_click_cpc REAL,
                last_year_cost_per_click_cpc REAL,
                seven_day_total_orders INTEGER,
                total_acos REAL,
                total_roas REAL,
                sales REAL,
                estimated_missed_sales_min REAL,
                estimated_missed_sales_max REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (start_date_key) REFERENCES Dim_Date(date_key),
                FOREIGN KEY (end_date_key) REFERENCES Dim_Date(date_key),
                FOREIGN KEY (campaign_key) REFERENCES Dim_Campaign(campaign_key),
                FOREIGN KEY (country_key) REFERENCES Dim_Country(country_key),
                FOREIGN KEY (currency_key) REFERENCES Dim_Currency(currency_key)
            );
            
            -- Create indexes for better query performance
            CREATE INDEX IF NOT EXISTS idx_fact_perf_date ON Fact_SP_Performance(date_key);
            CREATE INDEX IF NOT EXISTS idx_fact_perf_campaign ON Fact_SP_Performance(campaign_key);
            CREATE INDEX IF NOT EXISTS idx_fact_budget_campaign ON Fact_SP_Budget(campaign_key);
            CREATE INDEX IF NOT EXISTS idx_fact_budget_dates ON Fact_SP_Budget(start_date_key, end_date_key);
        """)
        
        self.conn.commit()
        print("Schema created successfully!")
    
    def get_or_create_date_key(self, date_str):
        """Get or create date dimension key"""
        if pd.isna(date_str):
            return None
            
        if date_str in self.dim_cache['date']:
            return self.dim_cache['date'][date_str]
        
        try:
            date_obj = pd.to_datetime(date_str)
            date_key = int(date_obj.strftime('%Y%m%d'))
            
            self.cursor.execute("""
                INSERT OR IGNORE INTO Dim_Date 
                (date_key, full_date, year, quarter, month, month_name, week, day_of_week, day_of_month, is_weekend)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                date_key,
                date_obj.date(),
                date_obj.year,
                (date_obj.month - 1) // 3 + 1,
                date_obj.month,
                date_obj.strftime('%B'),
                date_obj.isocalendar()[1],
                date_obj.weekday(),
                date_obj.day,
                date_obj.weekday() >= 5
            ))
            
            self.dim_cache['date'][date_str] = date_key
            return date_key
        except:
            return None
    
    def get_or_create_dimension(self, table, key_col, value_col, value, additional_cols=None):
        """Generic function to get or create dimension key"""
        if pd.isna(value):
            return None
        
        value = str(value).strip()
        cache_key = f"{value}_{additional_cols}" if additional_cols else value
        
        if cache_key in self.dim_cache.get(table.replace('Dim_', '').lower(), {}):
            return self.dim_cache[table.replace('Dim_', '').lower()][cache_key]
        
        # Try to find existing
        if additional_cols:
            self.cursor.execute(f"SELECT {key_col} FROM {table} WHERE unique_hash = ?", (cache_key,))
        else:
            self.cursor.execute(f"SELECT {key_col} FROM {table} WHERE {value_col} = ?", (value,))
        
        result = self.cursor.fetchone()
        
        if result:
            key = result[0]
        else:
            # Insert new
            if additional_cols:
                cols = list(additional_cols.keys()) + [value_col, 'unique_hash']
                vals = list(additional_cols.values()) + [value, cache_key]
                placeholders = ','.join(['?' for _ in vals])
                self.cursor.execute(
                    f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders})",
                    vals
                )
            else:
                self.cursor.execute(f"INSERT INTO {table} ({value_col}) VALUES (?)", (value,))
            
            key = self.cursor.lastrowid
        
        self.dim_cache[table.replace('Dim_', '').lower()][cache_key] = key
        return key
    
    def load_search_terms(self, csv_path):
        """Load SP search terms data"""
        print(f"Loading search terms from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        for idx, row in df.iterrows():
            date_key = self.get_or_create_date_key(row.get('date'))
            campaign_key = self.get_or_create_dimension('Dim_Campaign', 'campaign_key', 'campaign_name', row.get('campaign'),
                                                        {'portfolio_name': row.get('portfolio_name')})
            ad_group_key = self.get_or_create_dimension('Dim_Ad_Group', 'ad_group_key', 'ad_group_name', row.get('ad_group'))
            targeting_key = self.get_or_create_dimension('Dim_Targeting', 'targeting_key', 'targeting_value', row.get('targeting'),
                                                         {'match_type': row.get('match_type')})
            search_term_key = self.get_or_create_dimension('Dim_Search_Term', 'search_term_key', 'search_term_value', row.get('search_term'))
            retailer_key = self.get_or_create_dimension('Dim_Retailer', 'retailer_key', 'retailer_name', row.get('retailer'))
            country_key = self.get_or_create_dimension('Dim_Country', 'country_key', 'country_code', row.get('country'))
            currency_key = self.get_or_create_dimension('Dim_Currency', 'currency_key', 'currency_code', row.get('currency'))
            
            self.cursor.execute("""
                INSERT INTO Fact_SP_Performance (
                    date_key, campaign_key, ad_group_key, targeting_key, search_term_key,
                    retailer_key, country_key, currency_key,
                    impressions, clicks, click_thru_rate_ctr, cost_per_click_cpc, spend, sales,
                    total_acos, total_roas, seven_day_total_orders, seven_day_total_units,
                    seven_day_conversion_rate, seven_day_advertised_sku_units, seven_day_other_sku_units,
                    seven_day_advertised_sku_sales, seven_day_other_sku_sales
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                date_key, campaign_key, ad_group_key, targeting_key, search_term_key,
                retailer_key, country_key, currency_key,
                row.get('impressions'), row.get('clicks'), row.get('click-thru_rate_(ctr)'),
                row.get('cost_per_click_(cpc)'), row.get('spend'), row.get('sales'),
                row.get('total_advertising_cost_of_sales_(acos)'), row.get('total_return_on_advertising_spend_(roas)'),
                row.get('7_day_total_orders_(#)'), row.get('7_day_total_units_(#)'),
                row.get('7_day_conversion_rate'), row.get('7_day_advertised_sku_units_(#)'),
                row.get('7_day_other_sku_units_(#)'), row.get('7_day_advertised_sku_sales'),
                row.get('7_day_other_sku_sales')
            ))
            
            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1} rows...")
                self.conn.commit()
        
        self.conn.commit()
        print(f"Loaded {len(df)} search term records!")
    
    def load_budgets(self, csv_path):
        """Load SP budgets data"""
        print(f"Loading budgets from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        for idx, row in df.iterrows():
            start_date_key = self.get_or_create_date_key(row.get('start_date'))
            end_date_key = self.get_or_create_date_key(row.get('end_date'))
            campaign_key = self.get_or_create_dimension('Dim_Campaign', 'campaign_key', 'campaign_name', row.get('campaign'),
                                                        {'portfolio_name': row.get('portfolio_name'),
                                                         'program_type': row.get('program_type'),
                                                         'targeting_type': row.get('targeting_type'),
                                                         'bidding_strategy': row.get('bidding_strategy'),
                                                         'status': row.get('status')})
            country_key = self.get_or_create_dimension('Dim_Country', 'country_key', 'country_code', row.get('country'))
            currency_key = self.get_or_create_dimension('Dim_Currency', 'currency_key', 'currency_code', row.get('currency'))
            
            self.cursor.execute("""
                INSERT INTO Fact_SP_Budget (
                    start_date_key, end_date_key, campaign_key, country_key, currency_key,
                    budget, recommended_budget, average_time_in_budget,
                    impressions, last_year_impressions, estimated_missed_impressions_min, estimated_missed_impressions_max,
                    clicks, last_year_clicks, estimated_missed_clicks_min, estimated_missed_clicks_max,
                    click_thru_rate_ctr, spend, last_year_spend, cost_per_click_cpc, last_year_cost_per_click_cpc,
                    seven_day_total_orders, total_acos, total_roas, sales,
                    estimated_missed_sales_min, estimated_missed_sales_max
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                start_date_key, end_date_key, campaign_key, country_key, currency_key,
                row.get('budget'), row.get('recommended_budget'), row.get('average_time_in_budget'),
                row.get('impressions'), row.get('last_year_impressions'),
                row.get('estimated_missed_impressions_range_min'), row.get('estimated_missed_impressions_range_max'),
                row.get('clicks'), row.get('last_year_clicks'),
                row.get('estimated_missed_clicks_range_min'), row.get('estimated_missed_clicks_range_max'),
                row.get('click-thru_rate_(ctr)'), row.get('spend'), row.get('last_year_spend'),
                row.get('cost_per_click_(cpc)'), row.get('last_year_cost_per_click_(cpc)'),
                row.get('7_day_total_orders_(#)'), row.get('total_advertising_cost_of_sales_(acos)'),
                row.get('total_return_on_advertising_spend_(roas)'), row.get('sales'),
                row.get('estimated_missed_sales_range_min'), row.get('estimated_missed_sales_range_max')
            ))
            
            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1} rows...")
                self.conn.commit()
        
        self.conn.commit()
        print(f"Loaded {len(df)} budget records!")
    
    def load_campaigns(self, csv_path):
        """Load SP campaigns data"""
        print(f"Loading campaigns from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        for idx, row in df.iterrows():
            start_date_key = self.get_or_create_date_key(row.get('start_date'))
            end_date_key = self.get_or_create_date_key(row.get('end_date'))
            campaign_key = self.get_or_create_dimension('Dim_Campaign', 'campaign_key', 'campaign_name', row.get('campaign'),
                                                        {'portfolio_name': row.get('portfolio_name'),
                                                         'program_type': row.get('program_type'),
                                                         'targeting_type': row.get('targeting_type'),
                                                         'bidding_strategy': row.get('bidding_strategy'),
                                                         'status': row.get('status')})
            retailer_key = self.get_or_create_dimension('Dim_Retailer', 'retailer_key', 'retailer_name', row.get('retailer'))
            country_key = self.get_or_create_dimension('Dim_Country', 'country_key', 'country_code', row.get('country'))
            currency_key = self.get_or_create_dimension('Dim_Currency', 'currency_key', 'currency_code', row.get('currency'))
            
            # For campaigns, we'll insert into Fact_SP_Performance with campaign-level aggregation
            self.cursor.execute("""
                INSERT INTO Fact_SP_Performance (
                    date_key, campaign_key, retailer_key, country_key, currency_key,
                    impressions, clicks, click_thru_rate_ctr, cost_per_click_cpc, spend, sales,
                    total_acos, total_roas, seven_day_total_orders
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                start_date_key, campaign_key, retailer_key, country_key, currency_key,
                row.get('impressions'), row.get('clicks'), row.get('click-thru_rate_(ctr)'),
                row.get('cost_per_click_(cpc)'), row.get('spend'), row.get('sales'),
                row.get('total_advertising_cost_of_sales_(acos)'),
                row.get('total_return_on_advertising_spend_(roas)'),
                row.get('7_day_total_orders_(#)')
            ))
            
            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1} rows...")
                self.conn.commit()
        
        self.conn.commit()
        print(f"Loaded {len(df)} campaign records!")
    
    def load_placement(self, csv_path):
        """Load SP placement data"""
        print(f"Loading placement from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        for idx, row in df.iterrows():
            start_date_key = self.get_or_create_date_key(row.get('start_date'))
            end_date_key = self.get_or_create_date_key(row.get('end_date'))
            campaign_key = self.get_or_create_dimension('Dim_Campaign', 'campaign_key', 'campaign_name', row.get('campaign'),
                                                        {'portfolio_name': row.get('portfolio_name'),
                                                         'bidding_strategy': row.get('bidding_strategy')})
            placement_key = self.get_or_create_dimension('Dim_Placement', 'placement_key', 'placement_name', row.get('placement'))
            retailer_key = self.get_or_create_dimension('Dim_Retailer', 'retailer_key', 'retailer_name', row.get('retailer'))
            country_key = self.get_or_create_dimension('Dim_Country', 'country_key', 'country_code', row.get('country'))
            currency_key = self.get_or_create_dimension('Dim_Currency', 'currency_key', 'currency_code', row.get('currency'))
            
            self.cursor.execute("""
                INSERT INTO Fact_SP_Performance (
                    date_key, campaign_key, placement_key, retailer_key, country_key, currency_key,
                    impressions, clicks, cost_per_click_cpc, spend, sales,
                    total_acos, total_roas, seven_day_total_orders, seven_day_total_units
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                start_date_key, campaign_key, placement_key, retailer_key, country_key, currency_key,
                row.get('impressions'), row.get('clicks'), row.get('cost_per_click_(cpc)'),
                row.get('spend'), row.get('sales'),
                row.get('total_advertising_cost_of_sales_(acos)'),
                row.get('total_return_on_advertising_spend_(roas)'),
                row.get('7_day_total_orders_(#)'), row.get('7_day_total_units_(#)')
            ))
            
            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1} rows...")
                self.conn.commit()
        
        self.conn.commit()
        print(f"Loaded {len(df)} placement records!")
    
    def load_targeting(self, csv_path):
        """Load SP targeting data"""
        print(f"Loading targeting from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        for idx, row in df.iterrows():
            date_key = self.get_or_create_date_key(row.get('date'))
            campaign_key = self.get_or_create_dimension('Dim_Campaign', 'campaign_key', 'campaign_name', row.get('campaign'),
                                                        {'portfolio_name': row.get('portfolio_name')})
            ad_group_key = self.get_or_create_dimension('Dim_Ad_Group', 'ad_group_key', 'ad_group_name', row.get('ad_group'))
            targeting_key = self.get_or_create_dimension('Dim_Targeting', 'targeting_key', 'targeting_value', row.get('targeting'),
                                                         {'match_type': row.get('match_type')})
            retailer_key = self.get_or_create_dimension('Dim_Retailer', 'retailer_key', 'retailer_name', row.get('retailer'))
            country_key = self.get_or_create_dimension('Dim_Country', 'country_key', 'country_code', row.get('country'))
            currency_key = self.get_or_create_dimension('Dim_Currency', 'currency_key', 'currency_code', row.get('currency'))
            
            self.cursor.execute("""
                INSERT INTO Fact_SP_Performance (
                    date_key, campaign_key, ad_group_key, targeting_key, retailer_key, country_key, currency_key,
                    impressions, top_of_search_impression_share, clicks, click_thru_rate_ctr,
                    cost_per_click_cpc, spend, total_acos, total_roas, sales,
                    seven_day_total_orders, seven_day_total_units, seven_day_conversion_rate,
                    seven_day_advertised_sku_units, seven_day_other_sku_units,
                    seven_day_advertised_sku_sales, seven_day_other_sku_sales
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                date_key, campaign_key, ad_group_key, targeting_key, retailer_key, country_key, currency_key,
                row.get('impressions'), row.get('top-of-search_impression_share'), row.get('clicks'),
                row.get('click-thru_rate_(ctr)'), row.get('cost_per_click_(cpc)'), row.get('spend'),
                row.get('total_advertising_cost_of_sales_(acos)'),
                row.get('total_return_on_advertising_spend_(roas)'), row.get('sales'),
                row.get('7_day_total_orders_(#)'), row.get('7_day_total_units_(#)'),
                row.get('7_day_conversion_rate'), row.get('7_day_advertised_sku_units_(#)'),
                row.get('7_day_other_sku_units_(#)'), row.get('7_day_advertised_sku_sales'),
                row.get('7_day_other_sku_sales')
            ))
            
            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1} rows...")
                self.conn.commit()
        
        self.conn.commit()
        print(f"Loaded {len(df)} targeting records!")
    
    def close(self):
        """Close database connection"""
        self.conn.commit()
        self.conn.close()
        print(f"Database saved to {self.db_path}")


def main():
    """Main ETL execution"""
    print("=" * 60)
    print("Amazon SP Data Warehouse ETL")
    print("=" * 60)
    
    # Initialize ETL
    etl = SPDataWarehouseETL('sp_data_warehouse.db')
    
    try:
        # Create schema
        etl.create_schema()
        
        # Load data from CSV files
        # Update these paths to match your CSV file locations
        csv_files = {
            'search_terms': 'portfolio_split_output\Bedsheets_Sponsored_Products_Search_Term_Detailed_L30.csv',
            'budgets': 'portfolio_split_output\Bedsheets_Sponsored_Products_Budget_L30.csv',
            'campaigns': 'portfolio_split_output\Bedsheets_Sponsored_Products_Campaign_L30.csv',
            'placement': 'portfolio_split_output\Bedsheets_Sponsored_Products_Placement_L30.csv',
            'targeting': 'portfolio_split_output\Bedsheets_Sponsored_Products_Targeting_Detailed_L30.csv'
        }
        
        # Check which files exist and load them
        for file_type, file_path in csv_files.items():
            if Path(file_path).exists():
                if file_type == 'search_terms':
                    etl.load_search_terms(file_path)
                elif file_type == 'budgets':
                    etl.load_budgets(file_path)
                elif file_type == 'campaigns':
                    etl.load_campaigns(file_path)
                elif file_type == 'placement':
                    etl.load_placement(file_path)
                elif file_type == 'targeting':
                    etl.load_targeting(file_path)
            else:
                print(f"Warning: {file_path} not found, skipping...")
        
        print("\n" + "=" * 60)
        print("ETL Process Completed Successfully!")
        print("=" * 60)
        
        # Print summary statistics
        etl.cursor.execute("SELECT COUNT(*) FROM Fact_SP_Performance")
        perf_count = etl.cursor.fetchone()[0]
        
        etl.cursor.execute("SELECT COUNT(*) FROM Fact_SP_Budget")
        budget_count = etl.cursor.fetchone()[0]
        
        print(f"\nSummary:")
        print(f"  Performance Facts: {perf_count:,}")
        print(f"  Budget Facts: {budget_count:,}")
        print(f"  Campaigns: {len(etl.dim_cache['campaign']):,}")
        print(f"  Ad Groups: {len(etl.dim_cache['ad_group']):,}")
        print(f"  Search Terms: {len(etl.dim_cache['search_term']):,}")
        
    except Exception as e:
        print(f"\nError during ETL: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        etl.close()


if __name__ == "__main__":
    main()