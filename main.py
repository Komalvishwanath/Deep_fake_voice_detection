import os
import glob
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib

def extract_mfcc_features(audio_path, n_mfcc=13, n_fft=2048, hop_length=512):
    try:
        audio_data, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return None

    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return np.mean(mfccs.T, axis=0)

def create_dataset(directory, label):
    X, y = [], []
    audio_files = glob.glob(os.path.join(directory, "*.wav"))
    for audio_path in audio_files:
        mfcc_features = extract_mfcc_features(audio_path)
        if mfcc_features is not None:
            X.append(mfcc_features)
            y.append(label)
        else:
            print(f"Skipping audio file {audio_path}")

    print("Number of samples in", directory, ":", len(X))
    print("Filenames in", directory, ":", [os.path.basename(path) for path in audio_files])
    return X, y

def extract_audio_features(audio_file):
    # Load audio file
    y, sr = librosa.load(audio_file, sr=None)

    # Extract features
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    # Return features as a dictionary
    features = {
        "mfcc": mfcc,
        "chroma_stft": chroma_stft,
        "rms": rms,
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "rolloff": rolloff
    }

    return features


def train_model(X, y):
    X = np.array(X)
    y = np.array(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reshape features for LSTM input
    X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
    X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(128, input_shape=(X_train_scaled.shape[1], 1), return_sequences=True))
    model.add(LSTM(128))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train_reshaped, y_train, epochs=20, batch_size=32, validation_split=0.2)

    # Evaluate the model on test data
    loss, accuracy = model.evaluate(X_test_reshaped, y_test)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)

    # Save the trained model and scaler
    model.save("lstm_model.h5")
    scaler_filename = "scaler.pkl"
    joblib.dump(scaler, scaler_filename)

def analyze_audio(input_audio_path):
    model_filename = "lstm_model.h5"
    scaler_filename = "scaler.pkl"

    if not os.path.exists(input_audio_path):
        return "Error: The specified file does not exist."
    elif not input_audio_path.lower().endswith(".wav"):
        return "Error: The specified file is not a .wav file."

    mfcc_features = extract_mfcc_features(input_audio_path)
    if mfcc_features is not None:
        scaler = joblib.load(scaler_filename)
        mfcc_features_scaled = scaler.transform(mfcc_features.reshape(1, -1))

        # Reshape MFCC features to match LSTM input shape
        mfcc_features_reshaped = mfcc_features_scaled.reshape((1, mfcc_features_scaled.shape[0], 1))

        model = load_model(model_filename)
        prediction = model.predict(mfcc_features_reshaped)

        # Convert probability to class
        predicted_class = 1 if prediction[0][0] >= 0.5 else 0

        if predicted_class == 0:
            return "The input audio is classified as genuine."
        else:
            return "The input audio is classified as deepfake."
    else:
        return "Error: Unable to process the input audio."

def main():
    genuine_dir = r"real_all"
    deepfake_dir = r"fake_all"

    X_genuine, y_genuine = create_dataset(genuine_dir, label=0)
    X_deepfake, y_deepfake = create_dataset(deepfake_dir, label=1)

    # Combine datasets and shuffle
    X = X_genuine + X_deepfake
    y = y_genuine + y_deepfake
    X, y = shuffle(X, y)

    train_model(X, y)

if __name__ == "__main__":
    main()
