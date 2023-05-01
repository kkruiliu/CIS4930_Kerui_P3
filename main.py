

import os
import shutil
import random
import glob
import IPython.display as ipd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data_path = "D:/CIS4930_P3"
emotions = ["sad", "angry", "happy", "fear"]

# Step 1: Split the dataset into training and testing sets
train_data = []
test_data = []

for emotion in emotions:
    emotion_files = glob.glob(os.path.join(data_path, emotion, "*.wav"))
    random.shuffle(emotion_files)

    train_samples = emotion_files[:70]
    test_samples = emotion_files[70:]

    for sample in train_samples:
        train_data.append((sample, emotion))

    for sample in test_samples:
        test_data.append((sample, emotion))

#  Save the training and testing sets as separate files
train_df = pd.DataFrame(train_data, columns=["filepath", "emotion"])
test_df = pd.DataFrame(test_data, columns=["filepath", "emotion"])

train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)

#  Load the dataset and explore label distribution
train_df = pd.read_csv("train_data.csv")

# Analyze label distribution
label_counts = train_df["emotion"].value_counts()
print(label_counts)

# Plot label distribution
sns.barplot(x=label_counts.index, y=label_counts.values)
plt.title("Label Distribution")
plt.ylabel("Number of Samples")
plt.xlabel("Emotion")
plt.show()

#  Step 2:
#  Select sample audio files for each emotion
sample_audio_files = {}

for emotion in emotions:
    emotion_file = train_df[train_df["emotion"] == emotion]["filepath"].iloc[0]
    sample_audio_files[emotion] = emotion_file

# Listen to the sample audio files
for emotion, filepath in sample_audio_files.items():
    print(f"Emotion: {emotion}")
    display(ipd.Audio(filepath))

#  Plot the sample audio files in the time and frequency domain
def plot_audio_time_domain(filepath, title):
    signal, sample_rate = librosa.load(filepath)
    plt.figure(figsize=(10, 4))
    plt.plot(signal)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()

def plot_audio_frequency_domain(filepath, title):
    signal, sample_rate = librosa.load(filepath)
    spectrum = np.abs(librosa.stft(signal))
    log_spectrum = librosa.amplitude_to_db(spectrum, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_spectrum, sr=sample_rate, x_axis="time", y_axis="log")
    plt.title(title)
    plt.colorbar(format="%+2.0f dB")
    plt.show()

for emotion, filepath in sample_audio_files.items():
    plot_audio_time_domain(filepath, f"{emotion.capitalize()} - Time Domain")
    plot_audio_frequency_domain(filepath, f"{emotion.capitalize()} - Frequency Domain")

# Step 3: Acoustic Feature Extraction

def extract_features(filepath):
    signal, sample_rate = librosa.load(filepath)
    mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate)
    chroma_stft = librosa.feature.chroma_stft(y=signal, sr=sample_rate)
    spectral_contrast = librosa.feature.spectral_contrast(y=signal, sr=sample_rate)
    return mfccs, chroma_stft, spectral_contrast


train_df["features"] = train_df["filepath"].apply(extract_features)

# Step 4: Feature Post-processing

# Step 4.1: Feature Matrix Scaling
scaler = MinMaxScaler(feature_range=(-1, 1))

def scale_features(features):
    scaled_features = []
    for feature in features:
        scaled_feature = scaler.fit_transform(feature)
        scaled_features.append(scaled_feature)
    return tuple(scaled_features)

train_df["scaled_features"] = train_df["features"].apply(scale_features)

# Step 4.2: Feature Concatenation
def concatenate_features(scaled_features):
    return np.concatenate(scaled_features, axis=0)

train_df["concatenated_features"] = train_df["scaled_features"].apply(concatenate_features)

# Step 4.3: Feature Averaging
def average_features(concatenated_features):
    return np.mean(concatenated_features, axis=1)

train_df["averaged_features"] = train_df["concatenated_features"].apply(average_features)

# Prepare the training data
X_train = np.vstack(train_df["averaged_features"].values)
y_train = train_df["emotion"].values

# Prepare the testing data
test_df["features"] = test_df["filepath"].apply(extract_features)
test_df["scaled_features"] = test_df["features"].apply(scale_features)
test_df["concatenated_features"] = test_df["scaled_features"].apply(concatenate_features)
test_df["averaged_features"] = test_df["concatenated_features"].apply(average_features)

X_test = np.vstack(test_df["averaged_features"].values)
y_test = test_df["emotion"].values

# Step 5: Build the audio emotion recognition model

# Step 5.1: Prepare the data for training
X_train = np.array(train_df["averaged_features"].tolist())
y_train = train_df["emotion"].values

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Step 5.2: Train the model using different classifiers
classifiers = {
    "Support Vector Machine": SVC(),
    "Random Forest": RandomForestClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

trained_models = {}

for classifier_name, classifier in classifiers.items():
    model = classifier.fit(X_train, y_train_encoded)
    trained_models[classifier_name] = model

# Step 6: Model evaluation

# Step 6.1: Prepare the data for testing
test_df["features"] = test_df["filepath"].apply(extract_features)
test_df["scaled_features"] = test_df["features"].apply(scale_features)
test_df["concatenated_features"] = test_df["scaled_features"].apply(concatenate_features)
test_df["averaged_features"] = test_df["concatenated_features"].apply(average_features)

X_test = np.array(test_df["averaged_features"].tolist())
y_test = test_df["emotion"].values
y_test_encoded = label_encoder.transform(y_test)

# Step 6.2: Evaluate the performance of different classifiers
for classifier_name, model in trained_models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test_encoded, y_pred)
    print(f"Classifier: {classifier_name}")
    print(f"Accuracy: {accuracy}")
    print(classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_))
    print()
