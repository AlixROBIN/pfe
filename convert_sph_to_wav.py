import os
def verify_file_correspondence(wav_dir, stm_dir):
    wav_files = {os.path.splitext(f)[0] for f in os.listdir(wav_dir) if f.endswith(".wav")}
    stm_files = {os.path.splitext(f)[0] for f in os.listdir(stm_dir) if f.endswith(".stm")}

    missing_in_audio = stm_files - wav_files
    missing_in_transcriptions = wav_files - stm_files

    if missing_in_audio or missing_in_transcriptions:
        print(f"Manquent dans les audios : {missing_in_audio}")
        print(f"Manquent dans les transcriptions : {missing_in_transcriptions}")
    else:
        print("Tous les fichiers audio et transcriptions correspondent.")

# Vérifiez ici vos fichiers :
verify_file_correspondence("data/all/wav", "data/all/stm")
