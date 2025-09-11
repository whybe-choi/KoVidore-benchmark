#!/usr/bin/env python3
"""
Data setup script for KoVidore benchmark.
Downloads datasets from HuggingFace and organizes them into local structure.
"""

import logging
import os
import pandas as pd
from pathlib import Path
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dataset configurations
DATASETS = {
    "mir": {
        "path": "whybe-choi/kovidore-mir-v1.0-beir",
        "revision": "8a69ee34933fa2fb32ee5a20b1f537e99a22529f"
    },
    "vqa": {
        "path": "whybe-choi/kovidore-vqa-v1.0-beir", 
        "revision": "65771fa2bdd649c77faaab1b67378d5a782ae547"
    },
    "slide": {
        "path": "whybe-choi/kovidore-slide-v1.0-beir",
        "revision": "6d6c1727766d44434bb4a026724adbab67b41001"
    },
    "office": {
        "path": "whybe-choi/kovidore-office-v1.0-beir",
        "revision": "00faa811db01ab3dd1e5121fe60c85790aa93f42"
    },
    "finocr": {
        "path": "whybe-choi/kovidore-finocr-v1.0-beir",
        "revision": "0959e3bed492ecac92391ae01cbab1cd73725d28"
    }
}

def setup_dataset(subset_name: str, config: dict):
    """Setup a single dataset subset."""
    logger.info(f"Setting up {subset_name} dataset...")
    
    # Create directory structure
    subset_dir = Path(f"data/{subset_name}")
    subset_dir.mkdir(exist_ok=True)
    images_dir = subset_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Download different configs
    logger.info(f"Downloading {subset_name} from {config['path']}")
    
    # Load queries
    queries_dataset = load_dataset(
        config["path"], 
        "queries",
        split="test",
        revision=config["revision"]
    )
    
    # Load qrels
    qrels_dataset = load_dataset(
        config["path"], 
        "qrels",
        split="test",
        revision=config["revision"]
    )
    
    # Load corpus to get image info
    corpus_dataset = load_dataset(
        config["path"], 
        "corpus",
        split="corpus",
        revision=config["revision"]
    )
    
    # Prepare queries data
    queries_data = []
    for item in queries_dataset:
        queries_data.append({
            "query-id": item["_id"],
            "text": item["text"]
        })
    
    # Prepare qrels data
    qrels_data = []
    for item in qrels_dataset:
        query_id = item["query-id"]
        corpus_id = item["corpus-id"]
        score = item["score"]
        
        qrels_data.append({
            "query-id": query_id,
            "corpus-id": corpus_id,
            "score": score
        })
    
    # Save actual images from corpus
    for item in corpus_dataset:
        corpus_id = item["_id"]
        image_path = images_dir / f"{corpus_id}.jpg"
        if not image_path.exists() and "image" in item and item["image"] is not None:
            try:
                # Save the actual image
                item["image"].save(image_path, format="JPEG")
                logger.debug(f"Saved image {corpus_id}.jpg")
            except Exception as e:
                logger.warning(f"Failed to save image {corpus_id}: {e}")
                # Create empty placeholder if image save fails
                image_path.touch()
    
    # Save queries.csv
    queries_df = pd.DataFrame(queries_data)
    queries_df.to_csv(subset_dir / "queries.csv", index=False)
    logger.info(f"Saved {len(queries_data)} queries to {subset_dir}/queries.csv")
    
    # Save qrels.csv
    qrels_df = pd.DataFrame(qrels_data)
    qrels_df.to_csv(subset_dir / "qrels.csv", index=False)
    logger.info(f"Saved {len(qrels_data)} qrels to {subset_dir}/qrels.csv")
    
    image_count = len([f for f in images_dir.iterdir() if f.is_file()])
    logger.info(f"Saved {image_count} images in {images_dir}")
    logger.info(f"✅ {subset_name} setup completed!")

def main():
    """Main setup function."""
    logger.info("Starting KoVidore data setup...")
    
    # Ensure data directory exists
    Path("data").mkdir(exist_ok=True)
    
    for subset_name, config in DATASETS.items():
        try:
            setup_dataset(subset_name, config)
        except Exception as e:
            logger.error(f"Failed to setup {subset_name}: {e}")
            continue
    
    logger.info("Data setup completed!")
    logger.info("✅ Images have been downloaded and saved locally for testing.")

if __name__ == "__main__":
    main()