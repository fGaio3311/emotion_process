import librosa
import numpy as np
import pandas as pd
import os

# Mapeamento das emoções (baseado na documentação do RAVDESS)
emotion_map = {
    1: "neutral", 2: "neutral",  # Intensidade normal e forte para neutral
    3: "happy", 4: "sad", 5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"
}

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=22050, duration=3)  # Carrega 3 segundos
    audio = librosa.effects.trim(audio, top_db=20)[0]  # Remove silêncio
    
    # Extração de features
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13), axis=1)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr), axis=1)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr), axis=1)
    
    return np.hstack([mfccs, chroma, mel])

# Carregar dataset
features = []
labels = []

for root, dirs, files in os.walk("data/ravdess"):
    for file in files:
        if file.endswith(".wav"):
            emotion_code = int(file.split("-")[2])
            emotion = emotion_map.get(emotion_code, "unknown")
            
            file_path = os.path.join(root, file)
            feat = extract_features(file_path)
            
            features.append(feat)
            labels.append(emotion)

# Converter para DataFrame
df = pd.DataFrame(features)
df["label"] = labels

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Separar features e labels
X = df.drop("label", axis=1).values
y = df["label"].values

# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar SVM
model = SVC(kernel="rbf", C=10, gamma=0.01)
model.fit(X_train, y_train)

# Avaliar
y_pred = model.predict(X_test)
print(f"Acurácia: {accuracy_score(y_test, y_pred):.2f}")

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap="Blues")
plt.xticks(ticks=range(len(emotion_map.values())), labels=emotion_map.values(), rotation=45)
plt.yticks(ticks=range(len(emotion_map.values())), labels=emotion_map.values())
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.colorbar()
plt.show()

s
