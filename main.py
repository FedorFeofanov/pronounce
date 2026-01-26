from fastapi import FastAPI, UploadFile, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from starlette.responses import FileResponse 
from phone import wav_to_phone
from ml_phone import wav_to_IPA
import os
from pydub import AudioSegment
import librosa
import soundfile as sf
from fastapi.templating import Jinja2Templates
from models import Word
import database


app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get("/word/{id}", response_class=HTMLResponse)
async def read_index(id: str, request: Request):
    word: Word = database.get_word_by_id(id)
    context = {
        "request": request,
        "word": word.word,
        "ipa_uk": word.ipa_uk,
        "audio_uk": word.audio_uk,
        "ipa_us": word.ipa_us,
        "audio_us": word.audio_us,
    }
    return templates.TemplateResponse("index.html", context)

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
        result = wav_to_phone(wav_recording)
        print(f"Result: {result}")

    finally:
        for path in [raw_recording, wav_recording]:
            if os.path.exists(path):
                os.remove(path)
            
    return {"status": "success", "output": result}
