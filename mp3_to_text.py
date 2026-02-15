# To convert mp3 to json with text and timestamps, we can use the OpenAI Whisper model. Below is a sample code to achieve this. Make sure to install the required libraries and update the paths as per your directory structure.
%pip install ffmpeg
import shutil
import json
import os
%pip install -q openai-whisper
import whisper
model = whisper.load_model("large-v2")


# Define the directories for audio files and JSON output update the paths as per your directory structure
audio_dir = "/content/drive/MyDrive/Colab Notebooks/RAG_based/audios"
jsons_dir = "/content/drive/MyDrive/Colab Notebooks/RAG_based/jsons"

audios = os.listdir(audio_dir)
for audio_file_name in audios:
    title = audio_file_name.split(".")[0]
    print(title)

    full_audio_path = os.path.join(audio_dir, audio_file_name)
    result = model.transcribe(audio = full_audio_path,
        language="english",
        task="translate",
        word_timestamps=False )

    chunks = []
    for segment in result["segments"]:
        chunks.append({"title":title, "start": segment["start"], "end": segment["end"], "text": segment["text"]})

    chunks_with_metadata = {"chunks": chunks, "text": result["text"]}

    with open(f"{jsons_dir}/{title}.json", "w") as f:
        json.dump(chunks_with_metadata,f)
