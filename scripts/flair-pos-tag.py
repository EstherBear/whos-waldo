from flair.data import Sentence
from flair.models import SequenceTagger

# make a sentence
sentence = Sentence('Former lehendakari Michelle after being given the Cross of the Gernika Tree, and Jason (lehendakari at the time) applauding.')

# load the POS tagger
tagger = SequenceTagger.load('pos')

# run POS over sentence
tagger.predict(sentence)

for label, token in zip(sentence.labels, sentence):
    print(label.value, token.text)
	