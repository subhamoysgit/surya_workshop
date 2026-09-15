import logging
from pathlib import Path
from urllib.parse import urlparse
import pandas as pd
import s3fs

# Configure logger
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "download_progress.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

fs = s3fs.S3FileSystem(anon=True)

csv_path = Path("/d0/bjha/data/surya/workshop2026/indices/simsearch_surya_sharp_indices.csv")
surya_s3_index = pd.read_csv(csv_path)

total_entries = surya_s3_index.shape[0]
logger.info(f"Loaded index with {total_entries} entries.")

new_index = []

for i in range(total_entries):
    file_name = Path(surya_s3_index["path"][i]).relative_to(Path("/d0/bjha/data/surya/workshop2026/surya_ws_cache/"))
    file_name="s3://nasa-surya-bench/"+str(file_name)
    local_file_name = Path("/d0/bjha/data/surya/workshop2026/surya_ws_cache/") / Path(urlparse(file_name).path.lstrip("/"))
    local_file_name.parent.mkdir(parents=True, exist_ok=True)
    temp_dict = surya_s3_index.iloc[i].to_dict()
    temp_dict["path"] = local_file_name
    new_index.append(temp_dict)

    if Path(local_file_name).exists():
        logger.info(f"[{i+1}/{total_entries}] File already exists: {local_file_name}")
        continue

    try:
        fs.get(file_name, local_file_name)
        logger.info(f"[{i+1}/{total_entries}] Downloaded {file_name} to {local_file_name}")
    except Exception as e:
        logger.error(f"[{i+1}/{total_entries}] Failed to download {file_name}: {e}")

# output_csv = Path("data/indices/surya_subset_index_local.csv")
# output_csv.parent.mkdir(parents=True, exist_ok=True)
# pd.DataFrame(new_index).to_csv(output_csv, index=False)
# logger.info(f"Saved local index to {output_csv}")