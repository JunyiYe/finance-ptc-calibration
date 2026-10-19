#!/usr/bin/env python3
"""
Download evaluation datasets for real experiments.
"""

import os
from pathlib import Path
from datasets import load_dataset

def download_datasets():
    """Download all evaluation datasets."""
    
    data_dir = Path("data/evaluation_datasets")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    datasets_config = {
        "financial_phrasebank": {
            "name": "financial_phrasebank",
            "split": "train",
            "limit": 5000,
            "desc": "Financial sentiment analysis (PHRASEBANK)"
        },
        "conll2003": {
            "name": "conll2003", 
            "split": "train",
            "limit": 15000,
            "desc": "Named Entity Recognition (CONLL2003)"
        },
        "finqa": {
            "name": "finqa",
            "split": "train",
            "limit": 8000,
            "desc": "Financial QA (FinQA)"
        },
    }
    
    for key, config in datasets_config.items():
        print(f"\n{'='*70}")
        print(f"Downloading: {config['desc']}")
        print(f"{'='*70}")
        
        try:
            dataset = load_dataset(config["name"], split=config["split"], trust_remote_code=True)
            
            # Limit size
            if len(dataset) > config["limit"]:
                dataset = dataset.select(range(config["limit"]))
            
            # Save to disk
            save_path = data_dir / key
            dataset.save_to_disk(str(save_path))
            
            print(f"✓ Saved {len(dataset)} samples to {save_path}")
            print(f"  Size: {sum(f.stat().st_size for f in save_path.rglob('*') if f.is_file()) / 1e9:.2f} GB")
            
        except Exception as e:
            print(f"✗ Failed to download {key}: {e}")
            print(f"  Skipping...")

if __name__ == "__main__":
    download_datasets()
    print(f"\n{'='*70}")
    print("✓ All datasets ready!")
    print(f"{'='*70}")
