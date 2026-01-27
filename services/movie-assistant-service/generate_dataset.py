"""
Master script to run the complete dataset generation pipeline.
Executes: Data Collection → Wikipedia Enrichment → Feature Engineering → Dataset Generation
"""

import logging
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from data_collection.tmdb_collector import TMDBCollector
from data_collection.wikipedia_enricher import WikipediaEnricher
from data_collection.feature_engineering import FeatureEngineer
from dataset_generation.dataset_builder import DatasetBuilder
from data_collection.config import DATA_PATHS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run the complete pipeline."""
    logger.info("="*70)
    logger.info("MOVIE DISCOVERY ASSISTANT - DATASET GENERATION PIPELINE")
    logger.info("="*70)
    
    # Step 1: Data Collection from TMDB
    logger.info("\n[STEP 1/4] Collecting movie data from TMDB API...")
    if Path(DATA_PATHS["raw_movies"]).exists():
        logger.info(f"Found existing data at {DATA_PATHS['raw_movies']}. Skipping download.")
        with open(DATA_PATHS["raw_movies"], 'r', encoding='utf-8') as f:
            movies_data = json.load(f)
        
        # Convert keys to int if they are strings (JSON keys are always strings)
        if movies_data and isinstance(list(movies_data.keys())[0], str):
             movies_data = {int(k): v for k, v in movies_data.items()}
             
        logger.info(f"✓ Loaded {len(movies_data)} movies from disk")
    else:
        try:
            collector = TMDBCollector()
            movies_data = collector.collect_all_movies()
            collector.save_data(movies_data, DATA_PATHS["raw_movies"])
            logger.info(f"✓ Collected {len(movies_data)} movies from TMDB")
        except Exception as e:
            logger.error(f"✗ Data collection failed: {e}")
            return
    
    # Step 2: Wikipedia Enrichment (top 1000 movies)
    # Step 2: Wikipedia Enrichment
    logger.info("\n[STEP 2/4] Checking for Wikipedia enriched data...")
    enriched_path = DATA_PATHS["raw_movies"].replace(".json", "_enriched.json")
    
    if Path(enriched_path).exists():
        logger.info(f"✓ Found enriched data at {enriched_path}")
        try:
            with open(enriched_path, 'r', encoding='utf-8') as f:
                movies_data = json.load(f)
            
            # Ensure integer keys
            if movies_data and isinstance(list(movies_data.keys())[0], str):
                movies_data = {int(k): v for k, v in movies_data.items()}
                
            logger.info(f"✓ Loaded {len(movies_data)} enriched movies")
        except Exception as e:
            logger.error(f"✗ Failed to load enriched data: {e}")
            logger.info("Falling back to raw TMDB data.")
    else:
        logger.info("ℹ No enriched data found (movies_enriched.json).")
        logger.info("  To enrich: Run 'browser_wiki_enricher.html', download result, and save to data/raw/")
        logger.info("  Proceeding with raw TMDB data for now.")
    
    # Step 3: Feature Engineering
    logger.info("\n[STEP 3/4] Processing features...")
    try:
        engineer = FeatureEngineer()
        processed_features = engineer.process_all_movies(movies_data)
        engineer.save_features(processed_features, DATA_PATHS["processed_features"])
        logger.info(f"✓ Processed {len(processed_features)} movies with features")
    except Exception as e:
        logger.error(f"✗ Feature engineering failed: {e}")
        return
    
    # Step 4: Dataset Generation
    logger.info("\n[STEP 4/4] Generating training dataset...")
    try:
        builder = DatasetBuilder(processed_features)
        all_examples = builder.build_dataset()
        train_data, val_data, test_data = builder.split_dataset(all_examples)
        
        builder.save_dataset(train_data, DATA_PATHS["train_dataset"])
        builder.save_dataset(val_data, DATA_PATHS["val_dataset"])
        builder.save_dataset(test_data, DATA_PATHS["test_dataset"])
        
        logger.info(f"✓ Generated {len(all_examples)} total examples")
        logger.info(f"  - Train: {len(train_data)}")
        logger.info(f"  - Validation: {len(val_data)}")
        logger.info(f"  - Test: {len(test_data)}")
    except Exception as e:
        logger.error(f"✗ Dataset generation failed: {e}")
        return
    
    # Success!
    logger.info("\n" + "="*70)
    logger.info("✓ PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("="*70)
    logger.info("\nNext steps:")
    logger.info("1. Review the generated datasets in data/datasets/")
    logger.info("2. Upload train.jsonl, val.jsonl, test.jsonl to Google Colab")
    logger.info("3. Run the fine-tuning notebook: fine_tuning/llama_finetune_colab.ipynb")
    logger.info("\nDataset files:")
    logger.info(f"  - {DATA_PATHS['train_dataset']}")
    logger.info(f"  - {DATA_PATHS['val_dataset']}")
    logger.info(f"  - {DATA_PATHS['test_dataset']}")


if __name__ == "__main__":
    main()
