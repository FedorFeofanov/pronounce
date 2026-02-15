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
from g2p_en import G2p
from database import get_phoneme_vector

g2p = G2p()
MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
model2 = Wav2Vec2Model.from_pretrained(MODEL_ID)
#model.eval()
    

def wav_to_IPA(speech_array):
    input_values = processor(speech_array, return_tensors="pt", sampling_rate=16000).input_values

    with torch.no_grad():
        logits = model(input_values).logits # getting a table of probabilities

    predicted_ids = torch.argmax(logits, dim=-1) # picking the most likely answers
    transcription = processor.batch_decode(predicted_ids) # decoding tokens
    
    return transcription[0] # getting single transcription

def phone_to_vector(speech_array):
    # 1. Force audio to be float32 (Wav2Vec2 requirement)
    # if isinstance(speech_array, list):
    #     speech_array = np.array(speech_array).astype(np.float32)
        
    inputs = processor(speech_array, return_tensors="pt", sampling_rate=16000)
    print(inputs)

    with torch.no_grad():
        print("outputs")
        outputs = model2(inputs.input_values)
        # model(**inputs, output_hidden_states=True)
        print("all layers")
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
    data_rows = get_phoneme_vector(phoneme, word, number_of_rows, sex, vector)
    return data_rows[0][8] * 100
    

def word_to_phonemes(logits, word: str): # Split up word into phonemes
    print("phonemes")
    phonemes = [p for p in g2p(word) if p.strip()]
    print("tokens")
    tokens = processor.tokenizer.convert_tokens_to_ids(phonemes)
    print("log probs")
    log_probs = torch.log_softmax(logits, dim=-1)
    print("targets")
    targets = torch.tensor([tokens], dtype=torch.int32)
    
    # forced_align returns the best frame-by-frame path
    print("aligment")
    alignment, scores = forced_align(log_probs, targets)
    
    # 2. Group the path into segments
    # This logic identifies blocks of the same ID to get start/end
    segments = []
    current_token = alignment[0][0].item()
    start_frame = 0
    
    print("for loop")
    for i, frame_token in enumerate(alignment[0]):
        token_id = frame_token.item()
        if token_id != current_token:
            # End of previous phoneme, start of new one
            segments.append({
                "token": current_token,
                "start": start_frame,
                "end": i
            })
            start_frame = i
            current_token = token_id
            
    # Add the final leftover segment
    print("second append")
    segments.append({
        "token": current_token,
        "start": start_frame,
        "end": len(alignment[0])
    })
    
    print("returning")
    return segments

def phone_to_series(phone, speech_array):
    print("start")
    start = phone["start"]
    print(start)
    print("end")
    end = phone["end"]
    print(end)
    print(speech_array[start:end])
    return speech_array[start:end]

def score_recording(word, speech_array):
    print("getting logits")
    logits = get_logits(speech_array)
    # print("evaluating word corrctness")
    # if not validate_word(logits, speech_array, word): return None
    print("processing word into phonemes")
    phoneme_array = word_to_phonemes(logits, word)
    sum_scores = 0
    counter = 0
    print("evaluating phonemes")
    print(phoneme_array)
    for phone in phoneme_array:
        # print("checking if phoneme", phone, "empty")
        # if phone.label == '[pad]' or phone.label == '<pad>':
            # continue
        print("getting series")
        series = phone_to_series(phone, speech_array)
        print("getting vector")
        vector = phone_to_vector(series)
        print("comparing vectors")
        sum_scores += compare_phoneme_vec(vector, phone, word)
        counter += 1
    return sum_scores / counter if counter > 0 else 0

