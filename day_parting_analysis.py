import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================
# 1. LOAD DATA
# ============================
# Update this path
CSV_PATH = "SP_-_Campaign_-_Hourly_-_20_Oct_to_2nd_Nov.csv"
df = pd.read_csv(CSV_PATH)

# ============================
# 2. CLEAN DATA
# ============================
currency_cols = ["Spend", "7 Day Total Sales", "Budget"]

for col in currency_cols:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .fillna("0")
            .astype(float)
        )

df["Impressions"] = df["Impressions"].fillna(0)
df["Clicks"] = df["Clicks"].fillna(0)

# ============================
# 3. EXTRACT HOUR
# ============================
df["hour"] = (
    df["Start time"]
    .astype(str)
    .str.replace(":00", "", regex=False)
    .astype(int)
)

# ============================
# 4. AGGREGATE BY HOUR
# ============================
hourly = (
    df.groupby("hour")
    .agg(
        impressions=("Impressions", "sum"),
        clicks=("Clicks", "sum"),
        spend=("Spend", "sum"),
        sales=("7 Day Total Sales", "sum"),
    )
    .reset_index()
    .sort_values("hour")
)

# ============================
# 5. DERIVED METRICS
# ============================
hourly["CTR"] = np.where(
    hourly["impressions"] > 0,
    hourly["clicks"] / hourly["impressions"],
    0
)

hourly["CPC"] = np.where(
    hourly["clicks"] > 0,
    hourly["spend"] / hourly["clicks"],
    0
)

hourly["ROAS"] = np.where(
    hourly["spend"] > 0,
    hourly["sales"] / hourly["spend"],
    0
)

total_sales = hourly["sales"].sum()
hourly["sales_share"] = (
    hourly["sales"] / total_sales if total_sales > 0 else 0
)

# ============================
# 6. HOT / WARM / COLD LOGIC
# ============================
def classify_hour(row):
    if row["sales"] == 0 and row["spend"] > 0:
        return "COLD"
    if row["sales_share"] >= 0.15:
        return "HOT"
    if row["sales_share"] >= 0.05:
        return "WARM"
    return "COLD"

hourly["hour_type"] = hourly.apply(classify_hour, axis=1)

# ============================
# 7. SAVE OUTPUT
# ============================
hourly.to_csv("day_parting_analysis.csv", index=False)

print("\n=== DAY PARTING RESULTS ===\n")
print(hourly)

# ============================
# 8. VISUALIZATIONS
# ============================

# Impressions
plt.figure()
plt.bar(hourly["hour"], hourly["impressions"])
plt.xlabel("Hour of Day")
plt.ylabel("Impressions")
plt.title("Impressions by Hour")
plt.show()

# Clicks
plt.figure()
plt.bar(hourly["hour"], hourly["clicks"])
plt.xlabel("Hour of Day")
plt.ylabel("Clicks")
plt.title("Clicks by Hour")
plt.show()

# Spend
plt.figure()
plt.bar(hourly["hour"], hourly["spend"])
plt.xlabel("Hour of Day")
plt.ylabel("Spend")
plt.title("Spend by Hour")
plt.show()

# Sales (Key Day-Parting Chart)
plt.figure()
plt.bar(hourly["hour"], hourly["sales"])
plt.xlabel("Hour of Day")
plt.ylabel("Sales")
plt.title("Sales by Hour (Day Parting)")
plt.show()

# Sales Share (Normalised View)
plt.figure()
plt.bar(hourly["hour"], hourly["sales_share"])
plt.xlabel("Hour of Day")
plt.ylabel("Sales Share")
plt.title("Sales Share by Hour")
plt.show()
