"""
Quick Typesense setup: creates a collection per file in ./data using only the
first 5 rows to define schema and seed documents. Useful for a fast preview.

Usage:
  python typesense_setup_preview.py --api-key=xyz --host=localhost --port=8108 \
      --protocol=http --data-dir=./data --collection-prefix=preview --drop-existing
"""

import argparse
import os
import re
from pathlib import Path
from typing import Dict

import pandas as pd
import typesense


def snake_case(name: str) -> str:
    return re.sub(r"\W+", "_", name).strip("_").lower()


def infer_field_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_integer_dtype(series):
        return "int32"
    if pd.api.types.is_float_dtype(series):
        return "float"
    return "string"


def load_head(file_path: Path, n: int = 5) -> pd.DataFrame:
    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path, nrows=n)
    elif file_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(file_path, engine="openpyxl", nrows=n)
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")

    df.columns = [snake_case(c) for c in df.columns]
    df.insert(0, "id", [f"{file_path.stem}-{i}" for i in range(len(df))])
    df = df.convert_dtypes()
    return df


def build_schema(df: pd.DataFrame, collection_name: str) -> Dict:
    fields = [{"name": "id", "type": "string"}]
    for col in df.columns:
        if col == "id":
            continue
        fields.append({"name": col, "type": infer_field_type(df[col]), "facet": False, "optional": True})
    return {"name": collection_name, "enable_nested_fields": False, "fields": fields}


def ensure_collection(client: typesense.Client, schema: Dict, drop_existing: bool):
    name = schema["name"]
    try:
        client.collections[name].retrieve()
        if drop_existing:
            client.collections[name].delete()
    except Exception:
        pass
    try:
        client.collections.create(schema)
    except Exception:
        pass


def import_head(client: typesense.Client, df: pd.DataFrame, collection: str):
    records = df.to_dict(orient="records")
    if records:
        client.collections[collection].documents.import_(records, {"action": "upsert"})


def run_setup(api_key: str, host: str, port: int, protocol: str, data_dir: Path, collection_prefix: str, drop_existing: bool):
    client = typesense.Client(
        {
            "nodes": [{"host": host, "port": port, "protocol": protocol}],
            "api_key": api_key,
            "connection_timeout_seconds": 5,
        }
    )

    files = sorted([p for p in data_dir.iterdir() if p.suffix.lower() in {".csv", ".xlsx", ".xls"}])
    if not files:
        raise FileNotFoundError(f"No CSV/XLSX files found in {data_dir}")

    for file_path in files:
        df_head = load_head(file_path, n=5)
        collection_name = snake_case(f"{collection_prefix}_{file_path.stem}")
        schema = build_schema(df_head, collection_name)
        ensure_collection(client, schema, drop_existing=drop_existing)
        import_head(client, df_head, collection_name)
        print(f"✓ Created preview collection '{collection_name}' with {len(df_head)} rows from {file_path.name}")


def parse_args():
    parser = argparse.ArgumentParser(description="Create Typesense collections using first 5 rows of each data file.")
    parser.add_argument("--api-key", default=os.getenv("TYPESENSE_API_KEY"), required=False)
    parser.add_argument("--host", default=os.getenv("TYPESENSE_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("TYPESENSE_PORT", "8108")))
    parser.add_argument("--protocol", default=os.getenv("TYPESENSE_PROTOCOL", "http"))
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--collection-prefix", default="preview")
    parser.add_argument("--drop-existing", action="store_true", help="Drop and recreate collections before importing.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.api_key:
        raise SystemExit("Missing Typesense API key. Use --api-key or set TYPESENSE_API_KEY.")

    run_setup(
        api_key=args.api_key,
        host=args.host,
        port=args.port,
        protocol=args.protocol,
        data_dir=Path(args.data_dir),
        collection_prefix=args.collection_prefix,
        drop_existing=args.drop_existing,
    )
