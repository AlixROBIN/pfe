import os
import torch
from flask import Flask, request, jsonify, render_template, send_from_directory
from transformers import AutoProcessor, AutoModelForCTC
import torchaudio
from torchaudio.transforms import Resample
from time import sleep  # Simuler la progression

app = Flask(__name__)

# Répertoire des fichiers uploadés
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Charger le modèle et le processeur
processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Découper les fichiers audio longs en segments
def split_audio(file_path, segment_duration=10):
    waveform, sample_rate = torchaudio.load(file_path)
    num_samples_per_segment = int(segment_duration * sample_rate)

    segments = []
    for start in range(0, waveform.shape[1], num_samples_per_segment):
        end = start + num_samples_per_segment
        segments.append(waveform[:, start:end])

    return segments, sample_rate

# Rééchantillonner les fichiers audio à 16 kHz
def resample_audio(file_path, target_sample_rate=16000):
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != target_sample_rate:
        resampler = Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    return waveform, target_sample_rate

# Transcrire un fichier audio
@torch.no_grad()
def transcribe_audio(file_path):
    try:
        # Découper l'audio en segments
        segments, sample_rate = split_audio(file_path)
        transcription = []

        for i, segment in enumerate(segments):
            # Simuler la progression
            sleep(1)  # Simuler la progression de chaque segment (1 seconde par segment)

            # Prétraitement
            inputs = processor(segment.numpy(), sampling_rate=sample_rate, return_tensors="pt", padding=True)
            inputs = inputs.input_values.to(device)

            # Prédiction
            logits = model(inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)

            # Décodage
            transcription.append(processor.batch_decode(predicted_ids)[0])

        return " ".join(transcription)

    except Exception as e:
        return f"Erreur lors de la transcription : {str(e)}"

# Route principale
@app.route("/")
def index():
    return render_template("index.html")

# Route pour gérer le chargement de fichiers et la transcription
@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier n'a été envoyé."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nom de fichier invalide."}), 400

    if file:
        try:
            # Sauvegarde temporaire du fichier
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            # Conversion du fichier en WAV (si nécessaire)
            if not file.filename.endswith(".wav"):
                wav_file_path = os.path.splitext(file_path)[0] + ".wav"
                torchaudio.backend.sox_io_backend.info(file_path)  # Vérifie si le fichier est lisible
                waveform, sample_rate = torchaudio.load(file_path)
                torchaudio.save(wav_file_path, waveform, sample_rate)
                file_path = wav_file_path

            # Simuler la progression du traitement
            sleep(2)  # Simule le temps de conversion

            # Transcription
            transcription = transcribe_audio(file_path)

            return jsonify({"transcription": transcription}), 200

        except Exception as e:
            return jsonify({"error": f"Erreur lors du traitement du fichier : {str(e)}"}), 500

# Lancer l'application Flask
if __name__ == "__main__":
    app.run(debug=True)
