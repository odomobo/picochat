"""
The base/pretraining dataset is a set of parquet files.
This file contains utilities for:
- iterating over the parquet files and yielding documents from it
- download the files on demand if they are not on disk

For details of how the dataset was prepared, see `repackage_data_reference.py`.
"""

import os
import argparse
import time
import requests
import pyarrow.parquet as pq
from multiprocessing import Pool

from nanochat.common import get_data_dir

# -----------------------------------------------------------------------------
# The specifics of the current pretraining dataset

# The URL on the internet where the data is hosted and downloaded from on demand
BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
MAX_SHARD = 1822 # the last datashard is shard_01822.parquet
index_to_filename = lambda index: f"shard_{index:05d}.parquet" # format of the filenames
data_dir = get_data_dir()
BASE_DATA_DIR = os.path.join(data_dir, "base_data")
os.makedirs(BASE_DATA_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported

def list_parquet_files(corpus):
    """
    Looks into a corpus subdirectory and returns full paths to all parquet files.

    Args:
        corpus: Name of the corpus subdirectory in base_data/

    Returns:
        List of full paths to parquet files in the corpus directory
    """
    corpus_dir = os.path.join(BASE_DATA_DIR, corpus)
    if not os.path.exists(corpus_dir):
        raise ValueError(f"Corpus directory not found: {corpus_dir}")

    parquet_files = sorted([
        f for f in os.listdir(corpus_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])

    if not parquet_files:
        raise ValueError(f"No parquet files found in corpus directory: {corpus_dir}")

    parquet_paths = [os.path.join(corpus_dir, f) for f in parquet_files]
    return parquet_paths

def parquets_iter_batched(split, start=0, step=1, corpus=None):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". the last parquet file will be val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    - corpus: name of the corpus subdirectory in base_data/ (required)
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    if corpus is None:
        raise ValueError("corpus parameter is required")

    parquet_paths = list_parquet_files(corpus=corpus)
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts

# -----------------------------------------------------------------------------
def download_single_file(index):
    """ Downloads a single file index, with some backoff """

    # Construct the local filepath for this file and skip if it already exists
    filename = index_to_filename(index)
    filepath = os.path.join(BASE_DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    # Construct the remote URL for this file
    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")

    # Download with retries
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            # Write to temporary file first
            temp_path = filepath + f".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
            # Move temp file to final location
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            # Clean up any partial files
            for path in [filepath + f".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            # Try a few times with exponential backoff: 2^attempt seconds
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FineWeb-Edu 100BT dataset shards")
    parser.add_argument("-n", "--num-files", type=int, default=-1, help="Number of shards to download (default: -1), -1 = disable")
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of parallel download workers (default: 4)")
    args = parser.parse_args()

    num = MAX_SHARD + 1 if args.num_files == -1 else min(args.num_files, MAX_SHARD + 1)
    ids_to_download = list(range(num))
    print(f"Downloading {len(ids_to_download)} shards using {args.num_workers} workers...")
    print(f"Target directory: {BASE_DATA_DIR}")
    print()
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(download_single_file, ids_to_download)

    # Report results
    successful = sum(1 for success in results if success)
    print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {BASE_DATA_DIR}")
