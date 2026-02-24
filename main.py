from fastapi import FastAPI, UploadFile, Request, Form, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from starlette.responses import FileResponse 
from ml_phone import score_recording
import os
from pydub import AudioSegment
import librosa
import soundfile as sf
from fastapi.templating import Jinja2Templates
from models import Word
import database
import noisereduce as nr
import logging

logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force = True,
)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/{sex}/{word}", response_class=HTMLResponse)
async def read_index(request: Request, sex: str, word: str):
    is_valid_word = database.is_valid_word(word, sex.upper())
    if not is_valid_word: return f"<html><h1>we don't know the word {word} yet, sorry!<h1><html>"
    context = {
        "request": request,
        "word": word,
        #"ipa_uk": word.ipa_uk,
        #"audio_uk": word.audio_uk,
        #"ipa_us": word.ipa_us,
        #"audio_us": word.audio_us,
    }
    return templates.TemplateResponse("index.html", context)


@app.post("/post_audio") #endpoint for getting audio
async def get_audio(recording: UploadFile = File(...), sex: str = Form(...), word: str = Form(...)):
    logging.info(f"Post received for the word {word}")
    raw_recording = "./raw_recording.tmp"
    score = 0

    recording = await recording.read() # getting the recording

    with open(raw_recording, "wb") as temp_file:
        temp_file.write(recording) # writing the recording into a temp file
    raw_speech_array,_ = librosa.load(raw_recording, sr=16000) # converting audio to speech array
    reduced_noise_speech_array = nr.reduce_noise(y=raw_speech_array, sr=16000) # reducing noise
    speech_array, index = librosa.effects.trim(reduced_noise_speech_array, top_db=20) # removing silent parts
    
    logging.info("trying to get the score")
    try:
        score = score_recording(speech_array, sex.upper(), word)
        logging.info(f"Result: {score}")

    except:
        return {"status": "error", "output": ""}

    finally:
        for path in [raw_recording]:
            if os.path.exists(path):
                os.remove(path) # removing temp files
        return {"status": "success", "output": score}

