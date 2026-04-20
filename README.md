# Audio Emotion Recognition

A deep learning project that classifies human emotions from speech audio. It extracts acoustic features from audio files and trains a neural network to recognize 8 emotions: **angry, calm, disgust, fear, happy, neutral, sad, surprise**.

---

## Project Structure

```
.
├── audio_emotion_recognition.ipynb   # Feature extraction & EDA
├── classifier.ipynb                  # Model training & evaluation
├── data_path.csv                     # CSV mapping audio file paths to emotion labels
└── features.csv                      # Pre-extracted features (output of notebook 1)
```

---

## Workflow

### Notebook 1 — `audio_emotion_recognition.ipynb`

**Exploratory Data Analysis**
- Loads `data_path.csv` with audio file paths and emotion labels
- Plots emotion class distribution
- Visualizes waveplots and mel spectrograms for each emotion

**Data Augmentation**
Applies three augmentation techniques to improve model generalization:
- `noise()` — adds random Gaussian noise
- `stretch()` — time-stretches the audio (rate=0.8)
- `shift()` — randomly shifts the audio signal
- `pitch()` — shifts the pitch (factor=0.7)

**Feature Extraction**
Extracts MFCC features (20 coefficients) per audio clip using `librosa`. Audio is loaded with a 2.5s duration and 0.6s offset to skip silence. Features are saved to `features.csv`.

---

### Notebook 2 — `classifier.ipynb` (Google Colab)

**Preprocessing**
- Loads `features.csv` from Google Drive
- One-hot encodes emotion labels
- Splits data: 70% train / 18% validation / 12% test
- Applies `StandardScaler` normalization

**Model Architecture**
A 5-layer fully connected neural network (MLP):

| Layer | Units | Activation | Dropout |
|-------|-------|------------|---------|
| Input | 256 | ReLU | 0.2 |
| Hidden | 512 | ReLU | 0.2 |
| Hidden | 512 | ReLU | 0.2 |
| Hidden | 256 | ReLU | 0.2 |
| Hidden | 128 | ReLU | 0.2 |
| Output | 8 | Softmax | — |

**Training**
- Loss: `categorical_crossentropy`
- Optimizer: `Adam`
- Epochs: 500, Batch size: 32
- Best model saved via `ModelCheckpoint`

**Evaluation**
- Training/validation accuracy and loss curves
- Confusion matrix on test set

---

## Setup & Requirements

```bash
pip install librosa soundfile scikit-learn keras tensorflow pandas matplotlib seaborn numpy
```

For `classifier.ipynb`, it is designed to run on **Google Colab** with data stored in Google Drive.

---

## Usage

1. Prepare a `data_path.csv` with columns `Path` and `Emotions` pointing to your audio files.
2. Run `audio_emotion_recognition.ipynb` end-to-end to generate `features.csv`.
3. Upload `features.csv` to Google Drive and run `classifier.ipynb` on Colab to train the model.
4. The best model weights are saved as `audio_trial_proper.hdf5`.

---

## Dataset

The notebooks reference a dataset with 8 emotion classes. Update `data_path.csv` to point to your local audio files.
