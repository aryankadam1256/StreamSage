# Movie Discovery Assistant - Dataset & Fine-tuning

This directory contains the complete pipeline for creating a movie discovery dataset and fine-tuning Llama 3.2-3B for personalized movie recommendations.

## 📋 Overview

The pipeline consists of three main stages:

1. **Data Collection**: Fetch movie data from TMDB API
2. **Feature Engineering**: Process and enrich movie metadata
3. **Dataset Generation**: Create instruction-response pairs for fine-tuning
4. **Fine-tuning**: Train Llama 3.2-3B on Google Colab

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt
```

### Step 1: Generate Dataset

Run the complete pipeline:

```bash
python generate_dataset.py
```

This will:
- Collect ~5,000+ movies from TMDB
- Extract and engineer features
- Generate ~4,000 training examples
- Split into train/val/test sets

**Estimated Time**: 3-4 hours (due to API rate limits)

### Step 2: Fine-tune on Google Colab

1. Upload the generated datasets to Google Colab:
   - `data/datasets/train.jsonl`
   - `data/datasets/val.jsonl`
   - `data/datasets/test.jsonl`

2. Open `fine_tuning/llama_finetune_colab.ipynb` in Google Colab

3. Select GPU runtime (T4 is sufficient)

4. Run all cells

**Training Time**: ~2-3 hours on T4 GPU

## 📁 Project Structure

```
movie-assistant-service/
├── data_collection/
│   ├── config.py              # API keys and configuration
│   ├── tmdb_collector.py      # TMDB data collection
│   └── feature_engineering.py # Feature extraction
├── dataset_generation/
│   ├── query_templates.py     # Query templates
│   ├── response_generator.py  # Response generation
│   └── dataset_builder.py     # Dataset orchestration
├── fine_tuning/
│   └── llama_finetune_colab.ipynb  # Colab fine-tuning notebook
├── data/
│   ├── raw/                   # Raw TMDB data
│   ├── processed/             # Processed features
│   └── datasets/              # Final training datasets
└── generate_dataset.py        # Master pipeline script
```

## 🎯 Supported Query Types

The dataset supports 6 types of movie discovery queries:

1. **Similarity**: "Movies like Inception"
2. **Cast/Crew**: "Movies with Tom Hanks", "Christopher Nolan films"
3. **Genre**: "Sci-fi thriller", "Romantic comedy"
4. **Runtime**: "Movies under 2 hours", "Epic films"
5. **Mood**: "Uplifting movies", "Dark and gritty"
6. **Complex**: "Sci-fi movie under 2 hours with a twist"

## 🔧 Configuration

Edit `data_collection/config.py` to customize:

- **Collection Parameters**:
  - `num_popular_movies`: Number of popular movies to collect
  - `num_per_genre`: Movies per genre
  - `min_vote_count`: Quality filter

- **Dataset Parameters**:
  - `total_examples`: Target number of training examples
  - `query_type_distribution`: Distribution across query types
  - `train_split`, `val_split`, `test_split`: Dataset splits

## 🤖 Model Options

The Colab notebook is configured for **Llama 3.2-3B** (recommended for T4 GPU).

**Alternative models** (change `MODEL_NAME` in notebook):

| Model | Parameters | Training Time (T4) | Quality |
|-------|-----------|-------------------|---------|
| **Llama 3.2-3B** | 3B | ~2-3 hours | ⭐⭐⭐⭐ |
| Mistral 7B v0.3 | 7B | ~3-4 hours | ⭐⭐⭐⭐⭐ |
| Phi-3-Mini | 3.8B | ~2 hours | ⭐⭐⭐ |
| Gemma 2B | 2B | ~1-2 hours | ⭐⭐⭐ |

## 📊 Dataset Format

Training examples use the instruction format:

```json
{
  "instruction": "I loved Inception. What else should I watch?",
  "input": "",
  "output": "Based on Inception's mind-bending plot, I recommend:\n\n1. **Shutter Island** (2010) - Psychological thriller, 2h 18m\n   Leonardo DiCaprio in another reality-bending thriller...\n\n2. **Interstellar** (2014) - Sci-fi epic, 2h 49m\n   Christopher Nolan's space odyssey with complex themes..."
}
```

## 🎓 Features Extracted

For each movie, we extract:

- **Basic**: Title, year, runtime, genres, overview
- **Cast/Crew**: Director, top 10 cast, character names
- **Content**: Keywords, plot summary, tagline
- **Derived**: Mood tags, style tags, runtime category, era tag
- **Similarity**: TMDB recommendations, genre vectors
- **Ratings**: Vote average, popularity score

## 🔍 Example Queries & Responses

**Query**: "I need something uplifting and feel-good"

**Response**:
```
Here are some heartwarming, uplifting films:

1. **The Pursuit of Happyness** (2006) - Drama, 1h 57m
   Inspiring true story of perseverance and hope

2. **Paddington 2** (2017) - Family/Comedy, 1h 43m
   Charming and delightful family film

3. **The Intouchables** (2011) - Comedy/Drama, 1h 52m
   Touching French film about friendship
```

## 🚨 Troubleshooting

### Data Collection Issues

**Problem**: API rate limit errors
**Solution**: Increase `rate_limit_delay` in `config.py`

**Problem**: Missing movies
**Solution**: Check `min_vote_count` filter, may be too restrictive

### Fine-tuning Issues

**Problem**: Out of memory on Colab
**Solution**: 
- Reduce `per_device_train_batch_size` to 2
- Increase `gradient_accumulation_steps` to 8
- Use Gemma 2B instead

**Problem**: Training too slow
**Solution**:
- Reduce `num_train_epochs` to 2
- Reduce dataset size
- Use smaller model

## 📝 Next Steps

After fine-tuning:

1. **Test the model** with diverse queries
2. **Integrate into API** (update `main.py`)
3. **Deploy** with Docker
4. **Monitor** performance and collect feedback
5. **Iterate** on dataset quality

## 🔑 API Keys

Required API keys (set in `data_collection/config.py`):

- **TMDB API Key**: For movie data collection
- **Hugging Face Token**: For model access (optional)

## 📚 Resources

- [TMDB API Documentation](https://developers.themoviedb.org/3)
- [Llama 3.2 Model Card](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [PEFT Documentation](https://huggingface.co/docs/peft)

## 📄 License

This project uses data from TMDB. Please review [TMDB's Terms of Use](https://www.themoviedb.org/terms-of-use).

---

**Happy Fine-tuning! 🎬🤖**
