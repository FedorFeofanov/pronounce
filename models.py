class Word():

    def __init__(
        self,
        id = 0,
        word = "",
        ipa_uk = "",
        audio_uk = "",
        ipa_us = "",
        audio_us = ""):
        self.id = id
        self.word = word
        self.ipa_uk = ipa_uk
        self.audio_uk = audio_uk
        self.ipa_us = ipa_us
        self.audio_us = audio_us


class Phoneme():

    def __init__(
        self,
        id = 0,
        word = "",
        phoneme = "",
        embedding = "",
        duration = "",
        sex = "",
        speaker_type = "",
        subset = ""):
        self.id = id
        self.word = word
        self.phoneme = phoneme
        self.embedding = embedding
        self.duration = duration
        self.sex = sex
        self.speaker_type = speaker_type
        self.subset = subset

