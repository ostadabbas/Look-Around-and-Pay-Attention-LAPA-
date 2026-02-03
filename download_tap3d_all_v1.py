#!/usr/bin/env python3
"""
Script to download all TAP3D-Vid CMU panoptic sequences using the URL pattern.
"""

import os
import argparse
import requests
import numpy as np
from tqdm import tqdm

# Base URL for TAP3D-Vid minival dataset
BASE_URL = "https://storage.googleapis.com/dm-tapnet/tapvid3d/release_files/minival_v1.0/"

# List of all CMU panoptic sequences by category
TAP3D_SEQUENCES = {
    "basketball": [
        "basketball_3.npz", "basketball_4.npz", "basketball_5.npz", 
        "basketball_6.npz", "basketball_9.npz", "basketball_13.npz", 
        "basketball_14.npz", "basketball_20.npz", "basketball_24.npz", 
        "basketball_29.npz"
    ],
    "softball": [
        "softball_2.npz", "softball_9.npz", "softball_14.npz", 
        "softball_19.npz", "softball_21.npz", "softball_23.npz", 
        "softball_25.npz"
    ],
    "tennis": [
        "tennis_2.npz", "tennis_4.npz", "tennis_5.npz", 
        "tennis_17.npz", "tennis_22.npz", "tennis_23.npz", 
        "tennis_26.npz", "tennis_28.npz"
    ],
    "football": [
        "football_1.npz", "football_3.npz", "football_7.npz", 
        "football_16.npz", "football_19.npz", "football_21.npz", 
        "football_22.npz", "football_29.npz"
    ],
    "juggle": [
        "juggle_4.npz", "juggle_5.npz", "juggle_7.npz", 
        "juggle_8.npz", "juggle_9.npz", "juggle_22.npz"
    ],
    "boxes": [
        "boxes_5.npz", "boxes_6.npz", "boxes_7.npz", 
        "boxes_11.npz", "boxes_12.npz", "boxes_17.npz", 
        "boxes_19.npz", "boxes_22.npz", "boxes_27.npz", 
        "boxes_28.npz", "boxes_29.npz"
    ]
}

def download_file(url, output_path):
    """
    Download a file from a URL to a local path.
    
    Args:
        url: URL to download from
        output_path: Path to save the downloaded file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Skip if file already exists
    if os.path.exists(output_path):
        print(f"File already exists: {output_path}")
        return
    
    # Download file
    print(f"Downloading {url} to {output_path}")
    response = requests.get(url, stream=True)
    
    # Check if request was successful
    if response.status_code != 200:
        print(f"Failed to download {url}: {response.status_code}")
        return
    
    # Get file size
    file_size = int(response.headers.get('content-length', 0))
    
    # Download with progress bar
    with open(output_path, 'wb') as f, tqdm(
        desc=os.path.basename(output_path),
        total=file_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

def download_tap3d_sequences(base_output_dir, categories=None):
    """
    Download TAP3D-Vid sequences for specified categories.
    
    Args:
        base_output_dir: Base directory to save the downloaded sequences
        categories: List of categories to download. If None, download all categories.
    """
    # If no categories specified, download all
    if categories is None:
        categories = list(TAP3D_SEQUENCES.keys())
    
    # Download each category
    for category in categories:
        if category not in TAP3D_SEQUENCES:
            print(f"Category {category} not found. Skipping.")
            continue
        
        # Create category directory
        category_dir = os.path.join(base_output_dir, f"tap3d_{category}")
        os.makedirs(category_dir, exist_ok=True)
        
        # Download each sequence in the category
        for sequence in TAP3D_SEQUENCES[category]:
            url = BASE_URL + sequence
            output_path = os.path.join(category_dir, sequence)
            download_file(url, output_path)

def main():
    parser = argparse.ArgumentParser(description="Download TAP3D-Vid sequences")
    parser.add_argument("--output_dir", type=str, default="./data",
                        help="Base directory to save the downloaded sequences")
    parser.add_argument("--categories", type=str, nargs="+", 
                        choices=list(TAP3D_SEQUENCES.keys()),
                        help="Categories to download. If not specified, download all.")
    
    args = parser.parse_args()
    
    # Download TAP3D-Vid sequences
    download_tap3d_sequences(args.output_dir, args.categories)
    
    print("Done!")

if __name__ == "__main__":
    main()
