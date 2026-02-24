from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
)
import torch
import torchaudio
from torchaudio.functional import forced_align
from g2p import make_g2p
from database import get_phoneme_vector, is_valid_phoneme
import os
from pydub import AudioSegment
import librosa
import database
import numpy as np
from phonecodes import phonecodes
import logging

logging.basicConfig(
    filename='ml_phone.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force = True,
)
logging.info("Loading g2p")
g2p = make_g2p('eng', 'eng-ipa')
MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"
logging.info("Loading processor")
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
logging.info("Loading tokenizer")
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(MODEL_ID)
logging.info("Loading model")
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)


def phone_to_vector(speech_array):
    if len(speech_array) < 400:
        print("Warning: Slice is too short for Wav2Vec2. Padding to 400 samples.")
        # Pad with zeros to meet the minimum CNN requirement
        speech_array = np.pad(speech_array, (0, 400 - len(speech_array)), 'constant')

    inputs = processor(speech_array, return_tensors="pt", sampling_rate=16000)

    with torch.no_grad():
        outputs = model(inputs.input_values, output_hidden_states=True)
        all_layers = outputs.hidden_states 
        last_layer = all_layers[-1]
        vector = last_layer.mean(dim=1).squeeze().cpu().tolist()
    return vector

def get_logits(speech_array):
    inputs = processor(speech_array, return_tensors="pt", sampling_rate=16000)
    with torch.no_grad():
        return model(inputs.input_values).logits


def compare_phoneme_vec(vector, phoneme, sex, word): # Compare phoneme vector to those in the database
    number_of_rows = 1
    arpa_phone = phonecodes.ipa2arpabet(phoneme["phoneme"], "eng")
    if arpa_phone[-1].isdigit(): 
        arpa_phone = arpa_phone[:-1]
    if not is_valid_phoneme(arpa_phone, word): return 0
    data_rows = get_phoneme_vector(arpa_phone, word, number_of_rows, sex, vector)
    print("PHONEME")
    print(phoneme['phoneme'])
    print(arpa_phone)
    print(data_rows[0][8] * 100)
    if data_rows:
        return data_rows[0][8] * 100
    print("SOME ERROR IN POSTGRES..............")
    return 0
    

def word_to_phonemes(logits, word: str, total_samples: int): # Split up word into phonemes
    phonemes = processor.batch_decode(torch.argmax(logits, dim=-1))[0].split(" ")
    print(f"Phonemes: {phonemes}")

    tokens = processor.tokenizer.convert_tokens_to_ids(phonemes)
    log_probs = torch.log_softmax(logits, dim=-1)
    targets = torch.tensor([tokens], dtype=torch.int32)
    
    try:
        alignment, scores = forced_align(log_probs, targets)
    except Exception as e:
        print(f"Forced align error: {e}")
        raise
    STRIDE = 320
    segments = []
    current_token = alignment[0][0].item()
    start_frame = 0
    
    for i, frame_token in enumerate(alignment[0]):
        token_id = frame_token.item()
        if token_id != current_token:
            # End of previous phoneme, start of new one
            s_start = start_frame * STRIDE
            s_end = min(i * STRIDE, total_samples)
            segments.append({
                "phoneme": processor.tokenizer.convert_ids_to_tokens(current_token),
                "start": s_start,
                "end": s_end
            })
            start_frame = i
            current_token = token_id
            
    # Add the final leftover segment
    segments.append({
        "phoneme": processor.tokenizer.convert_ids_to_tokens(current_token),
        "start": start_frame * STRIDE,
        "end": total_samples
    })
    
    return segments

def phone_to_series(phone, speech_array):
    start = phone["start"]
    end = phone["end"]
    return speech_array[start:end]

def score_recording(speech_array, sex, word):
    logging.info("starting recording evaluation prepparation capitulation")
    logits = get_logits(speech_array)
    phoneme_array = word_to_phonemes(logits, word, len(speech_array))
    sum_scores = 0
    counter = 0
    logging.info(f"checking every single little smallest phoneme from {phoneme_array}")
    for phone in phoneme_array:
        if phone['phoneme'] == '<pad>': continue
        series = phone_to_series(phone, speech_array)
        logging.info(f"{phone}, {len(series)}")
        vector = phone_to_vector(series)
        vector_score = compare_phoneme_vec(vector, phone, sex, word)
        logging.info(f"Vector score {vector_score}")
        sum_scores += vector_score
        counter += 1
    return sum_scores / counter if counter > 0 else 0
