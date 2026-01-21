from transformers import (
    Wav2Vec2ForCTC, 
    Wav2Vec2Processor, 
    Wav2Vec2CTCTokenizer, 
    Wav2Vec2FeatureExtractor
)
import torch
import librosa

def wav_to_IPA(file_path: str):
    model_id = "facebook/wav2vec2-lv-60-espeak-cv-ft"
    
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_id)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
    
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    
    model = Wav2Vec2ForCTC.from_pretrained(model_id)

    # 3. Process the audio
    speech_array, sampling_rate = librosa.load(file_path, sr=16000)
    
    # Note: We use 'sampling_rate=16000' here to ensure the processor scales the audio correctly
    input_values = processor(speech_array, return_tensors="pt", sampling_rate=16000).input_values

    # 4. Inference
    with torch.no_grad():
        logits = model(input_values).logits

    # 5. Decode to IPA
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    
    return transcription[0] # batch_decode returns a list; we want the first string
