# Data Directory

This directory contains all local data for StreamSage:

- **chromadb/**: Vector database storage for subtitle embeddings
- **subtitles/**: Place your `.srt` subtitle files here
- **models/**: Trained model files from Google Colab
  - `binge_model.h5`: LSTM model for binge prediction
  - `sentiment_model/`: BERT model for sentiment analysis

## Getting Started

### 1. Add Subtitle Files

Place your movie subtitle files (`.srt` format) in the `subtitles/` directory:

```bash
data/subtitles/
├── inception.srt
├── the_matrix.srt
└── interstellar.srt
```

### 2. Ingest Subtitles

Run the ingestion script to populate the vector database:

```bash
docker exec -it streamsage-oracle python ingest.py --directory /app/data/subtitles
```

### 3. Add Trained Models

After training models on Google Colab, download and place them here:

```bash
data/models/
├── binge_model.h5           # From notebooks/train_rnn.ipynb
└── sentiment_model/          # From notebooks/train_bert.ipynb
    ├── config.json
    ├── pytorch_model.bin
    └── tokenizer files...
```

## Finding Subtitle Files

Free subtitle sources:
- [OpenSubtitles](https://www.opensubtitles.org/)
- [Subscene](https://subscene.com/)
- Movie DVD/Blu-ray extras

**Note**: Only use subtitles for movies you legally own.
