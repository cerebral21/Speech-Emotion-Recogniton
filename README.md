# Speech Emotion Recognition

Classifies the emotion in a short speech clip into one of eight classes (**angry, calm, disgust, fear, happy, neutral, sad, surprise**) from hand-engineered acoustic features and a fully connected neural network. Originally built in 2022 with librosa, scikit-learn and Keras.

**Result:** 83.0% accuracy on the validation split and **83.7% on the held-out test split**. These are optimistic for the reasons under [Limitations](#limitations): the split mixes augmented copies of the same clips and the same speakers across train and test.

---

## Data

12,162 labelled clips from four public acted-speech corpora (not included in this repo; download them from their publishers or from Kaggle):

| Corpus | Clips |
|---|---|
| CREMA-D | 7,442 |
| TESS | 2,800 |
| RAVDESS | 1,440 |
| SAVEE | 480 |

`data_path.csv` maps each file to its emotion label (columns `Emotions`, `Path`). The paths are from my machine (`G:/project/DATA/...`); change them to your own.

Class balance (clips, before augmentation):

| Emotion | Clips |
|---|---|
| happy, sad, angry, fear, disgust | 1,923 each |
| neutral | 1,703 |
| surprise | 652 |
| calm | 192 |

`calm` appears only in RAVDESS, so it is the smallest and the most corpus-specific class.

## Pipeline

### 1. Feature extraction (`audio emotion recognition.ipynb`)

Each clip is loaded for 2.5 seconds starting 0.6 seconds in (to skip leading silence). Each clip is averaged over time into a **191-dimensional feature vector**:

| Feature | Dimensions |
|---|---|
| Zero-crossing rate | 1 |
| Chroma (STFT) | 12 |
| Chroma (CENS) | 12 |
| Spectral centroid | 1 |
| Spectral bandwidth | 1 |
| Spectral contrast | 7 |
| Spectral flatness | 1 |
| Spectral roll-off | 1 |
| MFCC | 20 |
| RMS energy | 1 |
| Mel spectrogram | 128 |
| Tonnetz | 6 |

### 2. Augmentation

Every clip produces **four rows**: the original plus three augmented versions, giving 48,648 rows in total (12,162 x 4):

1. original
2. additive Gaussian noise
3. time-stretch (rate 0.8) followed by pitch shift
4. random time shift (up to about 5,000 samples)

### 3. Model (`classifier.ipynb`, run on Google Colab)

A fully connected network on the standardized 191 features:

`Dense 256 -> 512 -> 512 -> 256 -> 128 -> 8 (softmax)`, ReLU activations, dropout 0.2 after each hidden layer.

Training: categorical cross-entropy, Adam, batch size 32, 500 epochs, `ModelCheckpoint` saving the best validation loss. Labels are one-hot encoded and features standardized with a scaler fitted on the training split.

### 4. Splits

A random shuffled split with seed 1729:

| Split | Share | Rows | Use |
|---|---|---|---|
| Train | 70% | 34,053 | fitting |
| Validation | 12% | 5,838 | checkpoint selection during training |
| Test | 18% | 8,757 | final evaluation |

## Results

| Split | Accuracy |
|---|---|
| Validation (12%) | 83.0% |
| Test (18%) | 83.7% |

Both numbers come from the model after the final (500th) epoch, not from the best-validation-loss checkpoint. The notebook also plots training and validation curves and a confusion matrix for both splits. I did not compute per-class precision and recall.

## Limitations

- **Train/test leakage.** The split is random after augmentation, so augmented copies of the same recording can sit in both train and test. The same speakers also appear in all splits (TESS has only two), so the model can partly learn speakers rather than emotions. Treat the accuracy as an upper bound for a speaker-independent system.
- **No per-class metrics.** The classes are imbalanced (`calm` 192 clips, `surprise` 652), and overall accuracy hides weak classes. Use the confusion matrix and a per-class report.
- **Final-epoch weights.** The reported scores use the last-epoch model; the best-validation checkpoint was saved but not reloaded for evaluation.
- **Acted speech only.** All four corpora are recorded by actors in studio conditions, so results say little about spontaneous speech.
- **Library versions.** Written for librosa 0.8/0.9 and Keras 2.x on Colab in 2022. Newer librosa releases may need keyword arguments in `time_stretch`, `pitch_shift` and `melspectrogram`, and `waveplot` has been renamed.

## Next steps

- Split by speaker (or by corpus) and by source clip **before** augmenting, so test clips and their augmented copies never appear in training.
- Report per-class precision, recall and F1.
- Evaluate the best-validation checkpoint, not the final epoch.
- Compare against a baseline (logistic regression or gradient boosting on the same features) and a CNN on log-mel spectrograms.
- Add a small inference script that takes a `.wav` file and prints the predicted emotion.

## Running it

```bash
pip install librosa soundfile scikit-learn keras tensorflow pandas matplotlib seaborn numpy
```

1. Download the four corpora and update the paths in `data_path.csv`.
2. Run `audio emotion recognition.ipynb` to build `features.csv`.
3. Upload `features.csv` to Google Drive and run `classifier.ipynb` on Colab.
