from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model
)
import torch
import torchaudio
import torchaudio.functional as F
from torchaudio.functional import forced_align
from g2p import make_g2p
from database import get_phoneme_vector
import os
from pydub import AudioSegment
import librosa
import soundfile as sf
import database
import noisereduce as nr
import numpy as np
from phonecodes import phonecodes


def main():
    recording = "./think.wav"
    raw_speech_array,_ = librosa.load(recording, sr=16000)

    try:
        print("starting evaluating score")
        result = score_recording("because", raw_speech_array)
        print(f"Result: {result}")

    except Exception as e:
        print("ERROR", e)

g2p = make_g2p('eng', 'eng-ipa')
MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
model2 = Wav2Vec2Model.from_pretrained(MODEL_ID)
#model.eval()
    

def phone_to_vector(speech_array):
    if len(speech_array) < 400:
        print("Warning: Slice is too short for Wav2Vec2. Padding to 400 samples.")
        # Pad with zeros to meet the minimum CNN requirement
        speech_array = np.pad(speech_array, (0, 400 - len(speech_array)), 'constant')

    inputs = processor(speech_array, return_tensors="pt", sampling_rate=16000)

    with torch.no_grad():
        outputs = model2(inputs.input_values, output_hidden_states=True)
        all_layers = outputs.hidden_states 
        last_layer = all_layers[-1]
        vector = last_layer.mean(dim=1).squeeze().cpu().tolist()
    return vector

def get_logits(speech_array):
    inputs = processor(speech_array, return_tensors="pt", sampling_rate=16000)
    with torch.no_grad():
        return model(inputs.input_values).logits


def validate_word(logits, speech_array, word: str) -> bool:
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    print("The word that the user said", transcription)
    return word.lower() == transcription.lower()
    

def compare_phoneme_vec(vector, phoneme, word): # Compare phoneme vector to those in the database
    number_of_rows = 1
    sex = "M"
    arpa_phone = phonecodes.ipa2arpabet(phoneme["phoneme"], "eng")
    arpa_phone_clean = []
    for p in arpa_phone:
        if p[-1].isdigit(): p = p[:-1] # Remove stress numbers
        arpa_phone_clean.append(p)
    data_rows = get_phoneme_vector(arpa_phone, word, number_of_rows, sex, vector)
    print("PHONEME")
    print(phoneme['phoneme'])
    print(arpa_phone)
    if data_rows:
        return data_rows[0][8] * 100
    print("SOME ERROR IN POSTGRES..............")
    return 0
    

def word_to_phonemes(logits, word: str, total_samples: int): # Split up word into phonemes
    phonemes = list(g2p(word).output_string)
    print(f"Phonemes: {phonemes}")

    tokens = processor.tokenizer.convert_tokens_to_ids(phonemes)
    print(f"Tokens: {tokens}")
    print(f"Token count: {len(tokens)}")
    log_probs = torch.log_softmax(logits, dim=-1)
    print(f"Log probs shape: {log_probs.shape}")
    targets = torch.tensor([tokens], dtype=torch.int32)
    print(f"Targets shape: {targets.shape}")
    if -1 in tokens or 0 in tokens:
        print("WARNING: Unknown tokens detected!")
    
    try:
        alignment, scores = forced_align(log_probs, targets)
        print(f"Alignment type: {type(alignment)}")
        print(f"Alignment: {alignment}")
    except Exception as e:
        print(f"Forced align error: {e}")
        raise
    # alignment, scores = forced_align(log_probs, targets)
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

def score_recording(word, speech_array):
    logits = get_logits(speech_array)
    phoneme_array = word_to_phonemes(logits, word, len(speech_array))
    print("speech_array")
    print(len(speech_array))
    sum_scores = 0
    counter = 0
    for phone in phoneme_array:
        if phone['phoneme'] == '<pad>':
            continue
        series = phone_to_series(phone, speech_array)
        print(phone, len(series))
        vector = phone_to_vector(series)
        vector_score = compare_phoneme_vec(vector, phone, word)
        print(f"Vector score {vector_score}")
        sum_scores += vector_score
        counter += 1
    return sum_scores / len([p for p in phoneme_array if p['phoneme'] != "<pad>"])
    # return sum_scores / counter if counter > 0 else 0


if __name__ == "__main__":
    main()