from allosaurus.app import read_recognizer

def wav_to_phone(recording):
  model = read_recognizer("eng2102")
  return model.recognize(recording)
