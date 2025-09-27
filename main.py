import os
import librosa
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DATA_DIR = 'data'

# ----------------------------
# FEATURE EXTRACTION FUNCTION
# ----------------------------
def extract_features(file_path, duration=60):
    y, sr = librosa.load(file_path, duration=duration)

    # MFCC (13 values)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    # Chroma (12 values)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # Spectral contrast (7 values)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)

    # Tempo (force scalar)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_scalar = float(tempo)  # make sure it’s a single number
    tempo_arr = np.array([tempo_scalar])

    # Combine into one vector
    features = np.concatenate([mfcc_mean, chroma_mean, contrast_mean, tempo_arr])
    return features

# ----------------------------
# LOAD DATASET
# ----------------------------
def load_dataset():
    features, labels = [], []
    for genre in os.listdir(DATA_DIR):
        genre_path = os.path.join(DATA_DIR, genre)
        if os.path.isdir(genre_path):
            for file in os.listdir(genre_path):
                if file.endswith(".wav") or file.endswith(".mp3"):
                    path = os.path.join(genre_path, file)
                    try:
                        feat = extract_features(path)
                        if len(feat) != 33:   # sanity check
                            print(f"Skipping {file}, got {len(feat)} features (expected 33)")
                            continue
                        features.append(feat)
                        labels.append(genre)
                    except Exception as e:
                        print(f"Error with {file}: {e}")
                        continue
    return np.array(features), np.array(labels)

# ----------------------------
# MAIN PIPELINE
# ----------------------------
if __name__ == "__main__":
    # Load dataset
    X, y = load_dataset()

    print("Features shape:", X.shape)  # (#samples, #features)
    print("Labels shape:", y.shape)    # (#samples,)
    print("First 5 labels:", y[:5])

    # Split dataset (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Make a Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Test it
    y_pred = clf.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Predicted:", y_pred[:5])
    print("Actual:", y_test[:5])
