import os
import librosa
import numpy as np
from operator import itemgetter  # For sorting feature importance

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configuration
DATA_DIR = 'data'
SEGMENT_DURATION_S = 60  # Duration of each sample chunk in seconds
N_FEATURES_TO_SELECT = 20  # variable to limit features

# FEATURE EXTRACTION HELPER (69 Total Features)
def extract_segment_features(y, sr):
    """
    Extracts Mean AND Standard Deviation of MFCCs, Chroma, Contrast, RMS, ZCR, plus Tempo.
    Total features: 69
    """
    # 1. MFCC (13 values)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # 2. Chroma (12 values)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)

    # 3. Spectral contrast (7 values)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)
    contrast_std = np.std(contrast, axis=1)

    # 4. Root Mean Square (1 value) - Measures signal intensity/loudness
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)

    # 5. Zero Crossing Rate (ZCR) (1 value) - Measures percussive complexity/noisiness
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)

    # 6. Tempo (1 value)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_arr = np.array([float(tempo)])

    # Combine ALL features: Mean + Std Dev + Tempo (Total Features: 69)
    features = np.concatenate([
        mfcc_mean, mfcc_std,
        chroma_mean, chroma_std,
        contrast_mean, contrast_std,
        np.array([rms_mean, rms_std]), # Combining RMS mean and std
        np.array([zcr_mean, zcr_std]), # Combining ZCR mean and std
        tempo_arr
    ])
    return features

# LOAD DATASET
def load_dataset_segmented():
    """
    Loads full audio files and breaks them into 60-second segments.
    """
    features, labels = [], []
    print(f"Searching for data in: {DATA_DIR}")

    for genre in sorted(os.listdir(DATA_DIR)):
        genre_path = os.path.join(DATA_DIR, genre)
        if os.path.isdir(genre_path):
            count_files = 0
            count_segments = 0
            for file in os.listdir(genre_path):
                if file.endswith(".wav") or file.endswith(".mp3"):
                    path = os.path.join(genre_path, file)
                    count_files += 1

                    try:
                        y_full, sr = librosa.load(path)
                    except Exception as e:
                        print(f"Error loading {file}: {e}")
                        continue

                    segment_samples = sr * SEGMENT_DURATION_S

                    for i in range(0, len(y_full) - segment_samples + 1, segment_samples):
                        y_segment = y_full[i: i + segment_samples]

                        feat = extract_segment_features(y_segment, sr)

                        if len(feat) != 69:
                            continue

                        features.append(feat)
                        labels.append(genre)
                        count_segments += 1

            print(f"Loaded {count_segments} segments from {count_files} files for '{genre}'")

    return np.array(features), np.array(labels)

# MAIN PIPELINE
if __name__ == "__main__":
    #  1. Load Data and Encode Labels
    X, y = load_dataset_segmented()

    if len(X) == 0:
        print("\nERROR: No features loaded. Check your 'data' directory structure or file formats.")
        exit()

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print("\n--- Label Encoding Mapping ---")
    for i, label in enumerate(le.classes_):
        print(f"Class {i}: {label}")

    print("\nFeatures shape (Initial 65):", X.shape)
    print("Labels shape:", y_encoded.shape)

    # Split dataset (80% train, 20% test)
    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # 2. Feature Selection
    print("\n" + "=" * 60)
    print(f"      Feature Selection: Training RF to find Top {N_FEATURES_TO_SELECT}")
    print("=" * 60)

    # Use a temporary RF model to rank features
    temp_clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    temp_clf_rf.fit(X_train_full, y_train)

    # Get feature importances and sort them
    importances = temp_clf_rf.feature_importances_

    # Create a list of (index, importance) tuples and sort by importance
    feature_ranking = sorted(
        [(i, importance) for i, importance in enumerate(importances)],
        key=itemgetter(1),
        reverse=True
    )

    # Select indices of the top N features
    top_feature_indices = [item[0] for item in feature_ranking[:N_FEATURES_TO_SELECT]]

    # Filter the datasets to include only the top features
    X_train = X_train_full[:, top_feature_indices]
    X_test = X_test_full[:, top_feature_indices]

    print(f"Selected {N_FEATURES_TO_SELECT} features. New Training shape:", X_train.shape)

    # 3. Random Forest Classifier (The Main Model)
    print("\n" + "=" * 60)
    print(f"      Random Forest Classifier (Primary Model) - {N_FEATURES_TO_SELECT} Features")
    print("=" * 60)

    clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    # Train using filtered features
    clf_rf.fit(X_train, y_train)
    y_pred_rf = clf_rf.predict(X_test)

    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    print(f"Overall Random Forest Accuracy: {rf_accuracy:.4f}")

    # Classification Report
    report_rf = classification_report(
        y_test,
        y_pred_rf,
        target_names=le.classes_,
        zero_division=0
    )
    print("\nRandom Forest Classification Report:\n", report_rf)

    # Confusion Matrix
    conf_mat_rf = confusion_matrix(y_test, y_pred_rf)
    print("\nRandom Forest Confusion Matrix (Rows=Actual, Cols=Predicted):\n", conf_mat_rf)

    # 4. K-Nearest Neighbors (KNN) Comparison
    print("\n" + "=" * 60)
    print(f"     K-Nearest Neighbors (KNN) Classifier - {N_FEATURES_TO_SELECT} Features")
    print("=" * 60)

    knn_clf = KNeighborsClassifier(n_neighbors=5)
    # Train using filtered features
    knn_clf.fit(X_train, y_train)
    knn_y_pred = knn_clf.predict(X_test)

    knn_accuracy = accuracy_score(y_test, knn_y_pred)
    print(f"Overall KNN Accuracy: {knn_accuracy:.4f}")

    report_knn = classification_report(
        y_test,
        knn_y_pred,
        target_names=le.classes_,
        zero_division=0
    )
    print("\nKNN Classification Report:\n", report_knn)

    # 5. Final Summary
    print("\n" + "-" * 60)
    print(f"Final Summary: Random Forest Accuracy: {rf_accuracy:.4f} | KNN Accuracy: {knn_accuracy:.4f}")
    print("-" * 60)