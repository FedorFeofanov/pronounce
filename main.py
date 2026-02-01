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
import noisereduce as nr


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/word/{id}", response_class=HTMLResponse)
async def read_index(request: Request, id: str):
    word: Word = database.get_word_by_id(id)
    print(word)
    context = {
        "request": request,
        "word": word.word,
        "ipa_uk": word.ipa_uk,
        "audio_uk": word.audio_uk,
        "ipa_us": word.ipa_us,
        "audio_us": word.audio_us,
    }
    return templates.TemplateResponse("index.html", context)

@app.post("/post_audio") #endpoint for getting audio
async def get_audio(recording: UploadFile):
    raw_recording = "./raw_recording.tmp"
    result = ""

    content = await recording.read() # getting the recording

    with open(raw_recording, "wb") as temp_file:
        temp_file.write(content) # writing the recording into a temp file

    raw_speech_array,_ = librosa.load(raw_recording, sr=16000) # converting audio to speech array

    reduced_noise_speech_array = nr.reduce_noise(y=raw_speech_array, sr=16000) # reducing noise

    speech_array, index = librosa.effects.trim(reduced_noise_speech_array, top_db=20) # removing silent parts
    
    try:
        result = wav_to_IPA(speech_array) # converting speech array to IPA
        print(f"Result: {result}")

    except:
        return {"status": "error", "output": ""}

    finally:
        for path in [raw_recording]:
            if os.path.exists(path):
                os.remove(path) # removing temp files
        return {"status": "success", "output": result}

