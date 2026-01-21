from fastapi import FastAPI, UploadFile
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse 
from phone import wav_to_phone
from ml_phone import wav_to_IPA
import os
from pydub import AudioSegment
import librosa
import soundfile as sf


app = FastAPI()

@app.get("/")
async def read_index():
    return FileResponse('./static/index.html')

@app.post("/post_audio")
async def get_audio(recording: UploadFile):
    raw_recording = "./raw_recording.tmp"
    wav_recording = "./recording.wav"
    result = ""

    content = await recording.read()

    with open(raw_recording, "wb") as temp_file:
        temp_file.write(content)

    x,_ = librosa.load(raw_recording, sr=16000)
    sf.write(wav_recording, x, 16000, subtype="PCM_16")
    
    try:
        result = wav_to_IPA(wav_recording)
        print(f"Result: {result}")

    finally:
        for path in [raw_recording, wav_recording]:
            if os.path.exists(path):
                os.remove(path)
            
    return {"status": "success", "output": result}
