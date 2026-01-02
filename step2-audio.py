import os
from pydub import AudioSegment

def split_audio_to_folders(file_path, folder_2min, folder_3sec):
    try:
        audio = AudioSegment.from_file(file_path, format="wav")
        os.makedirs(folder_2min, exist_ok=True)
        os.makedirs(folder_3sec, exist_ok=True)

        chunk_duration_2min = 2 * 60 * 1000
        for i in range(0, len(audio), chunk_duration_2min):
            chunk = audio[i:i + chunk_duration_2min]
            chunk_filename = f"chunk_{i // chunk_duration_2min + 1}.wav"
            chunk.export(os.path.join(folder_2min, chunk_filename), format="wav")

        chunk_duration_3sec = 3 * 1000
        for i in range(0, len(audio), chunk_duration_3sec):
            chunk = audio[i:i + chunk_duration_3sec]
            chunk_filename = f"chunk_{i // chunk_duration_3sec + 1}.wav"
            chunk.export(os.path.join(folder_3sec, chunk_filename), format="wav")

        print("Audio splitting into 2-minute and 3-second chunks completed successfully.")
    except Exception as e:
        print(f"Error while splitting audio: {e}")

file_path = "C:\\EL-3rdsem\\STRESS\\ThirdAttempt\\webapp\\audiofile-step1\\merged_audio.wav"
folder_2min = "C:\\EL-3rdsem\\STRESS\\ThirdAttempt\\webapp\\audiofile-step3-2"
folder_3sec = "C:\\EL-3rdsem\\STRESS\\ThirdAttempt\\webapp\\audiofile-step3-3"

split_audio_to_folders(file_path, folder_2min, folder_3sec)
