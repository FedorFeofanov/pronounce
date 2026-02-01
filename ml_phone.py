from transformers import (
    Wav2Vec2ForCTC, 
    Wav2Vec2Processor, 
    Wav2Vec2CTCTokenizer, 
    Wav2Vec2FeatureExtractor
)
import torch

def wav_to_IPA(speech_array):
    model_id = "facebook/wav2vec2-lv-60-espeak-cv-ft"
    
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_id)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
    
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    
    model = Wav2Vec2ForCTC.from_pretrained(model_id)

    input_values = processor(speech_array, return_tensors="pt", sampling_rate=16000).input_values

    with torch.no_grad():
        logits = model(input_values).logits # getting a table of probabilities

    predicted_ids = torch.argmax(logits, dim=-1) # picking the most likely answers
    transcription = processor.batch_decode(predicted_ids) # decoding tokens
    
    return transcription[0] # getting single transcription
