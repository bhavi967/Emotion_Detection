# Emotion_Detection
import os
import librosa
import librosa.display
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
# Define a function to extract audio features
def extract_features(audio_path):
    y, sr = librosa.load(audio_path)
    # Extract relevant features (e.g., MFCCs, pitch)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    pitch, _ = librosa.piptrack(y=y, sr=sr)
    # Aggregate features
    features = np.mean(mfccs, axis=1)
    features = np.append(features, np.mean(pitch))
    return features
# Create a list to store features and corresponding labels
data = []
labels = []
# Process each audio file in the dataset
for emotion in ["happy", "sad"]:
    emotion_dir = os.path.join("mixkit-ending-show-audience-clapping-478” ,”mixkit-lost-kid-sobbing-474” ,emotion)
    for audio_file in os.listdir(emotion_dir):
        audio_path = os.path.join(emotion_dir, audio_file)
        features = extract_features(audio_path)
        data.append(features)
        labels.append(emotion)
# Convert lists to numpy arrays
X = np.array(data)
y = np.array(labels)
# Train a simple classifier (SVM)
classifier = SVC(kernel='linear')
classifier.fit(X, y)
# Create a DataFrame to store results
results = pd.DataFrame(columns=["Audio File", "Predicted Emotion"])
# Predict emotion for new audio files and store in the DataFrame
new_data = []
for audio_file in os.listdir("new_audio"):
    audio_path = os.path.join("new_audio", audio_file)
    features = extract_features(audio_path)
    new_data.append(features)
    predicted_emotion = classifier.predict([features])[0]
    results = results.append({"Audio File": , "Predicted Emotion": predicted_emotion}, ignore_index=True)
# Create a line chart
positive = (results["Predicted Emotion"] == "happy").astype(int)
negative = (results["Predicted Emotion"] == "sad").astype(int)
plt.plot(results["Audio File"], positive, label="Happy", marker='o', color='g')
plt.plot(results["Audio File"], -negative, label="Sad", marker='o', color='r')
plt.xlabel("Audio File")
plt.ylabel("Emotion")
plt.title("Emotion Prediction")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.show()
# Print the results as a table
print(results)

