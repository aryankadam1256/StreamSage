# Google Colab Training Notebooks

This directory will contain Jupyter notebooks for training models on Google Colab:

- **train_rnn.ipynb**: LSTM model for binge prediction (Phase 2)
- **train_bert.ipynb**: BERT model for sentiment analysis (Phase 3)

These notebooks will be created in the next phases of the project.

## Why Google Colab?

Training deep learning models requires:
- GPU acceleration (much faster than CPU)
- Significant RAM (8GB+ for BERT)
- Python environment with TensorFlow/PyTorch

Google Colab provides all of this for FREE, making it perfect for:
- Learning and experimentation
- Training models without expensive hardware
- Quick iteration cycles

## Workflow

1. Open notebook in Google Colab
2. Run all cells to train the model
3. Download the saved model files (.h5, .pt)
4. Place in `../data/models/`
5. Restart the appropriate Docker service

The local services only run **inference**, not training.
