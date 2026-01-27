# Alternative Dataset Generation - Using Pre-made TMDB Dataset

Since we're experiencing network connectivity issues with the TMDB API, we'll use a pre-collected dataset instead.

## Option 1: Download TMDB 5000 Movies Dataset

This is a well-known dataset available on Kaggle with 5000 movies including:
- Movie metadata (title, overview, genres, runtime, etc.)
- Cast and crew information
- Keywords
- Ratings and popularity

### Steps:

1. **Download the dataset:**
   - Go to: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
   - Download `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`
   - Place them in `data/raw/` directory

2. **Or use the direct download script:**
   ```bash
   # Install kaggle CLI
   pip install kaggle
   
   # Set up Kaggle API credentials (create at kaggle.com/account)
   # Place kaggle.json in ~/.kaggle/ or C:\Users\<username>\.kaggle\
   
   # Download dataset
   kaggle datasets download -d tmdb/tmdb-movie-metadata -p data/raw/ --unzip
   ```

3. **Run the alternative generation script:**
   ```bash
   python generate_dataset_from_csv.py
   ```

## Option 2: Use Hugging Face Dataset

Alternatively, we can use the TMDB dataset from Hugging Face:

```bash
pip install datasets
python generate_dataset_from_hf.py
```

## What We'll Generate

The alternative approach will:
1. Load the pre-made CSV/dataset
2. Process and clean the data
3. Extract features (genres, keywords, mood tags, etc.)
4. Generate 4,000 instruction-response pairs
5. Create train/val/test splits

**Estimated time:** 15-20 minutes (vs 3+ hours with API)

## Next Steps

Choose one of the options above and I'll create the corresponding generation script.
