import os
from flask import Flask, request, render_template
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
from tensorflow.keras.models import load_model


app = Flask(__name__)

def extract_mfcc_features(audio_path, n_mfcc=13, n_fft=2048, hop_length=512):
    try:
        audio_data, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return None

    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return np.mean(mfccs.T, axis=0)

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

        # Reshape MFCC features directly into the desired shape
        mfcc_features_reshaped = mfcc_features_scaled.reshape((1, 13, 1))

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


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "audio_file" not in request.files:
            return render_template("index.html", message="No file part")
        
        audio_file = request.files["audio_file"]
        if audio_file.filename == "":
            return render_template("index.html", message="No selected file")
        
        if audio_file and allowed_file(audio_file.filename):
            if not os.path.exists("uploads"):
                os.makedirs("uploads")
                
            audio_path = os.path.join("uploads", audio_file.filename)
            audio_file.save(audio_path)
            result = analyze_audio(audio_path)
            os.remove(audio_path) 
            return render_template("result.html", result=result)
        
        return render_template("index.html", message="Invalid file format. Only .wav files allowed.")
    
    return render_template("index.html")

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() == "wav"

if __name__ == "__main__":
    app.run(debug=True)
