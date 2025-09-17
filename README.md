# BMS-MFZS: A Background Modeling and Suppression Method based on Multi-Feature Zero-Shot Learning

This is a Python-based **background suppression system** designed for multi-modal zero-shot learning. The proposed method integrates visual, semantic, temporal, and spectral features into a unified framework to effectively suppress complex backgrounds and enhance the recognition of unseen target classes.

This repository includes all training code, testing code, and dataset placeholders used in the experiments. The proposed algorithm is the result of rigorous theoretical analysis and careful implementation. 

We commit to releasing all source code upon acceptance of the paper.

---

## 📁 Project Structure

```bash
BMS-MFZS/
├── image_feature_extraction.py         # Image feature extraction
├── text_feature_extraction.py          # Label semantic feature extraction
├── time_feature_extraction.py          # Temporal feature extraction
├── spectrum_feature_extraction.py      # Spectral feature extraction
├── transformer_network.py              # Transformer-based background modeling
├── data_gen.py                         # Dataset loader and batch generator
├── model.py                            # Zero-Shot model architecture
├── background_reconstruct.py           # Background reconstruction module
├── Zero-Shot-Background-Train.py       # Model training entry
├── Zero-Shot-Background-Test.py        # Model testing and background suppression
├── configs/                            # Optional configuration files
└── data/                               # Dataset directory (provided by user)



## Module Description and Workflow

### 1. Feature Extraction Modules

- `image_feature_extraction.py`: Extracts image features using pre-trained backbones.
- `text_feature_extraction.py`: Extracts semantic features of labels using BERT tokenizer or similar.
- `time_feature_extraction.py`: Encodes temporal dynamics between consecutive frames.
- `spectrum_feature_extraction.py`: Extracts multi-band spectral features from hyperspectral or multispectral data.

### 2. Background Modeling

- `transformer_network.py`: Builds a transformer-based background modeling network to learn background feature mapping functions.

### 3. Dataset Loader

- `data_gen.py`: Implements a PyTorch `CustomDataset` that returns multi-modal features (image, text, time, spectrum) and multi-label annotations in sliding windows. Used in both training and testing.

### 4. Model Definition

- `model.py`: Defines the core `ZeroShotModel`, which integrates all modalities and performs multi-label classification in a zero-shot setting.

  Key imports:
  ```python
  from image_feature_extraction import ImageFeatureExtractor
  from text_feature_extraction import TextFeatureExtractor
  from time_feature_extraction import TimeFeatureExtractor
  from spectrum_feature_extraction import SpectrumFeatureExtractor
  from transformer_network import TransformerNetwork
  ```

### 5. Training Script

 -  `Zero-Shot-Background-Train.py`: Initializes the model and dataset, trains the model with aligned multi-modal inputs, and saves checkpoints.

### 6. Background Reconstruction & Testing

 -  `background_reconstruct.py`: Contains the `reconstruct_background()` function that generates background representations using the learned background feature mappings.

 -  `Zero-Shot-Background-Test.py`: Loads the trained model, applies background reconstruction, performs local contrast enhancement, and classifies unseen classes. Local contrast suppression is integrated to improve detection under cluttered backgrounds.
