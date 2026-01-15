"""
Typesense ingestion helper for Amazon Ads datasets.

Reads CSV/XLSX files from a directory (default: ./data) and loads each file
into its own Typesense collection for fast lookup and search.

CLI example:
    python typesense_ingest.py --api-key=xyz --host=localhost --port=8108 \
        --protocol=http --data-dir=./data --collection-prefix=ads --drop-existing
"""

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
import typesense


def snake_case(name: str) -> str:
    """Convert arbitrary column/file names to snake_case."""
    return re.sub(r"\W+", "_", name).strip("_").lower()


def infer_field_type(series: pd.Series) -> str:
    """Map pandas dtype to Typesense field type."""
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_integer_dtype(series):
        return "int32"
    if pd.api.types.is_float_dtype(series):
        return "float"
    return "string"


def build_schema(df: pd.DataFrame, collection_name: str) -> Dict:
    """Create Typesense schema from dataframe columns."""
    fields = [{"name": "id", "type": "string"}]
    for col in df.columns:
        if col == "id":
            continue
        fields.append(
            {"name": col, "type": infer_field_type(df[col]), "facet": False, "optional": True}
        )

    return {
        "name": collection_name,
        "enable_nested_fields": False,
        "fields": fields,
    }


def clean_currency_value(val):
    """Strip currency symbols and convert to float."""
    if pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    cleaned = str(val).replace("$", "").replace(",", "").strip()
    if cleaned == "" or cleaned.lower() == "nan":
        return None
    try:
        return float(cleaned)
    except ValueError:
        return val  # Return original if not a number


def load_dataframe(file_path: Path) -> pd.DataFrame:
    """Load CSV or XLSX file into a cleaned dataframe."""
    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
    elif file_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(file_path, engine="openpyxl")
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")

    # Standardize column names
    df.columns = [snake_case(c) for c in df.columns]

    # Add stable id for Typesense
    df.insert(0, "id", [f"{file_path.stem}-{i}" for i in range(len(df))])

    # Clean currency columns (columns with $ values like budget_amount, spend, sales)
    currency_patterns = ["budget", "spend", "cost", "sales", "cpc", "price", "amount"]
    for col in df.columns:
        if any(pattern in col.lower() for pattern in currency_patterns):
            df[col] = df[col].apply(clean_currency_value)

    # Let pandas try sensible dtypes for inference
    df = df.convert_dtypes()
    return df


def chunk_records(records: List[Dict], size: int = 500):
    """Yield chunks of records for bulk import."""
    for i in range(0, len(records), size):
        yield records[i : i + size]


def ensure_collection(client: typesense.Client, schema: Dict, drop_existing: bool):
    """Create collection, optionally dropping existing one."""
    name = schema["name"]
    try:
        client.collections[name].retrieve()
        if drop_existing:
            client.collections[name].delete()
    except Exception:
        # Collection does not exist; safe to create
        pass

    try:
        client.collections.create(schema)
    except Exception:
        # Already exists and not dropped; continue
        pass


def import_dataframe(client: typesense.Client, df: pd.DataFrame, collection_name: str):
    """Import a dataframe into Typesense in chunks."""
    # Ensure all values are JSON-serializable (convert datetimes to ISO strings)
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str)

    records = df.to_dict(orient="records")
    for batch in chunk_records(records):
        client.collections[collection_name].documents.import_(batch, {"action": "upsert"})


def run_ingest(
    api_key: str,
    host: str,
    port: int,
    protocol: str,
    data_dir: Path,
    collection_prefix: str,
    drop_existing: bool,
):
    client = typesense.Client(
        {
            "nodes": [{"host": host, "port": port, "protocol": protocol}],
            "api_key": api_key,
            "connection_timeout_seconds": 5,
        }
    )

    files = sorted(
        [p for p in data_dir.iterdir() if p.suffix.lower() in {".csv", ".xlsx", ".xls"}]
    )
    if not files:
        raise FileNotFoundError(f"No CSV/XLSX files found in {data_dir}")

    for file_path in files:
        df = load_dataframe(file_path)
        collection_name = snake_case(f"{collection_prefix}_{file_path.stem}")
        schema = build_schema(df, collection_name)
        ensure_collection(client, schema, drop_existing=drop_existing)
        import_dataframe(client, df, collection_name)
        print(f"✓ Imported {len(df):,} rows from {file_path.name} into '{collection_name}'")


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest CSV/XLSX files into Typesense.")
    parser.add_argument("--api-key", default=os.getenv("TYPESENSE_API_KEY"), required=False)
    parser.add_argument("--host", default=os.getenv("TYPESENSE_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("TYPESENSE_PORT", "8108")))
    parser.add_argument("--protocol", default=os.getenv("TYPESENSE_PROTOCOL", "http"))
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--collection-prefix", default="amazon_ads")
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop and recreate collections before importing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.api_key:
        raise SystemExit("Missing Typesense API key. Use --api-key or set TYPESENSE_API_KEY.")

    run_ingest(
        api_key=args.api_key,
        host=args.host,
        port=args.port,
        protocol=args.protocol,
        data_dir=Path(args.data_dir),
        collection_prefix=args.collection_prefix,
        drop_existing=args.drop_existing,
    )
