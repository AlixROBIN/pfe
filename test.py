import os
import json
import torch
import torchaudio
from pydub import AudioSegment
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Charger le modèle et le processeur
model_name = "facebook/wav2vec2-large-960h-lv60-self"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Envoyer le modèle sur GPU si disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Fonction pour découper un fichier audio en morceaux de taille spécifiée
def split_audio(audio_path, chunk_length_ms=5000):
    audio = AudioSegment.from_file(audio_path)
    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    return chunks

# Fonction pour resampler un fichier audio
def resample_audio(audio_path, target_sample_rate=16000):
    speech_array, sampling_rate = torchaudio.load(audio_path)
    if sampling_rate != target_sample_rate:
        transform = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=target_sample_rate)
        speech_array = transform(speech_array)
    return speech_array.squeeze().numpy(), target_sample_rate

# Fonction pour transcrire un fichier audio
def transcribe_audio(audio_path):
    try:
        # Découpage de l'audio en morceaux de 5 secondes
        chunks = split_audio(audio_path, chunk_length_ms=5000)
        transcription = ""
        
        for i, chunk in enumerate(chunks):
            print(f"Traitement du morceau {i + 1}/{len(chunks)} de {audio_path}")
            
            # Sauvegarde temporaire du morceau pour le traiter
            chunk_path = f"temp_chunk_{i}.wav"
            chunk.export(chunk_path, format="wav")
            
            # Resampler et préparer les entrées
            speech_array, sampling_rate = resample_audio(chunk_path)
            inputs = processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
            input_values = inputs.input_values.to(device)
            
            # Faire la prédiction
            with torch.no_grad():
                logits = model(input_values).logits
            
            # Décodage
            predicted_ids = torch.argmax(logits, dim=-1)
            chunk_transcription = processor.batch_decode(predicted_ids)[0]
            transcription += chunk_transcription + " "
            
            # Supprimer le fichier temporaire
            os.remove(chunk_path)
        
        return transcription.strip()
    except Exception as e:
        print(f"Erreur lors de la transcription de {audio_path} : {e}")
        return f"Erreur : {e}"

# Fonction principale
def main():
    test_audio_dir = "data/test/wav"
    results = {}
    
    if not os.path.exists(test_audio_dir):
        raise FileNotFoundError(f"Le dossier '{test_audio_dir}' est introuvable.")
    
    for audio_file in os.listdir(test_audio_dir):
        if not audio_file.endswith('.wav'):
            continue
        
        audio_path = os.path.join(test_audio_dir, audio_file)
        print(f"Traitement de l'audio : {audio_path}")
        
        # Transcrire le fichier audio
        transcription = transcribe_audio(audio_path)
        results[audio_file] = transcription
        print(f"Transcription de {audio_file} : {transcription}")
        
        # Sauvegarde intermédiaire des résultats
        with open("results.json", "w") as f:
            json.dump(results, f, indent=4)
    
    print("\nTranscriptions terminées. Résultats sauvegardés dans 'results.json'.")

if __name__ == "__main__":
    main()
