from custom_predictor import Predictor
from flair.data import Sentence
from flair.models import SequenceTagger

tagger = SequenceTagger.load('pos')
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz", 'custom_semantic_role_labeling')

sentence = 'Secretary of Defense Ronald greets Margaret, Canadian Minister of National Defense at NATO Headquarters in Brussels, Belgium as they prepare to meet for a bilateral discussion to discuss matters of mutual importance. Ronald is participating in his first NATO ministerial as defense secretary June 24, 2015.'
flairSentence = Sentence(sentence)
tagger.predict(flairSentence)
labels = [label.value for label in flairSentence.labels]
tokens = [token.text for token in flairSentence]

result = predictor.predict_tokenized(tokenized_sentence=tokens, labels=labels)

print(result)
