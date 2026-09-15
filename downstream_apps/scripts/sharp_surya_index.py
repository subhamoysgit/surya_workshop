"""
sharp_surya_index.py

Generates an index CSV file for similarity search (simsearch) by extracting 
metadata from precomputed Surya embedding NetCDF files of SHARP active regions.

For each embedding file, two timestamp entries are recorded:
  1. 1 hour before reference time (t - 1h) -> matching 'surya_file_1h_before'
  2. Reference time (t)                   -> matching 'surya_file'

The resulting index dataframe is sorted chronologically and saved as a CSV.
"""

from pathlib import Path
import pandas as pd
import xarray
from urllib.parse import urlparse


# -----------------------------------------------------------------------------
# Configuration Paths
# -----------------------------------------------------------------------------
# Directory containing precomputed Surya embedding NetCDF files for SHARPs
sharp_embedding_path = Path("/d0/subhamoy/sharps/surya_embeddings")
surya_stack_path=Path("/d0/bjha/data/surya/workshop2026/surya_ws_cache/")

# Output destination for the generated similarity search index CSV
output_csv_path = Path("/d0/bjha/data/surya/workshop2026/indices/simsearch_surya_sharp_indices.csv")


# -----------------------------------------------------------------------------
# Extract Metadata from Embedding Files
# -----------------------------------------------------------------------------
# Find and sort all NetCDF embedding files recursively
files = sorted(sharp_embedding_path.rglob("*.nc"))
dataindex = []

for file in files:
    # Load dataset attributes without keeping the full array in memory
    sharp_embedding = xarray.load_dataset(file)
    attrs = sharp_embedding.attrs

    # Extract common bounding box and active region attributes
    x0 = attrs.get("x0")
    y0 = attrs.get("y0")
    x1 = attrs.get("x1")
    y1 = attrs.get("y1")
    harpnum = attrs.get("harpnum")
    ref_time = pd.to_datetime(attrs.get("reference_time"))
    embed_path_str = str(file)

    # 1. Entry for the solar state 1 hour prior to reference time (t - 1h)
    out_attrs_1h_before = {
        "timestep": ref_time - pd.to_timedelta("1h"),
        "path": str(surya_stack_path/Path(urlparse(attrs.get("surya_file_1h_before")).path.lstrip("/"))),
        "embed_path": embed_path_str,
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "present": 1,
        "harpnum": harpnum,
    }
    dataindex.append(out_attrs_1h_before)

    # 2. Entry for the solar state at the reference time (t)
    out_attrs_current = {
        "timestep": ref_time,
        "path": str(surya_stack_path/Path(urlparse(attrs.get("surya_file")).path.lstrip("/"))),
        "present": 1,
        "embed_path": embed_path_str,
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "harpnum": harpnum,
    }
    dataindex.append(out_attrs_current)


# -----------------------------------------------------------------------------
# Build and Save Index DataFrame
# -----------------------------------------------------------------------------
# Create DataFrame, sort chronologically by reference time, and reset index
df = pd.DataFrame(dataindex).sort_values(by="timestep").reset_index(drop=True)

# Ensure output directory exists before saving
output_csv_path.parent.mkdir(parents=True, exist_ok=True)

# Export the index to CSV
df.to_csv(output_csv_path, index=False)
 