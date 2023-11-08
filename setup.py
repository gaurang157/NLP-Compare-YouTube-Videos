# setup.py
# import nltk
from pysentimiento import create_analyzer

# nltk.download('punkt')
analyzer = create_analyzer(task="sentiment", lang="en")
emotion_analyzer = create_analyzer(task="emotion", lang="en")
hate_speech_analyzer = create_analyzer(task="hate_speech", lang="en")
