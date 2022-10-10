"""
Compile the list of examples from the Who's Waldo dataset such that:
- it's an image from test set
- the caption has an unseen verb
"""
popular_names = ['James', 'John', 'Robert', 'Michael', 'William', 'David', 'Richard', 'Charles', 'Joseph', 'Thomas', 
                'Christopher', 'Daniel', 'Paul', 'Mark', 'Donald', 'George', 'Kenneth', 'Steven', 'Edward', 'Brian', 
                'Ronald', 'Anthony', 'Kevin', 'Jason', 'Matthew', 'Gray', 'Timothy', 'Jose', 'Larry', 'Jeffrey']
sample = "President Jack meeting with Japanese Prime Minister Lucy."
sample = "Senator James meeting with President John."
import sys
sys.path.append('.')
import json
import os
import nltk.data
from flair.models import SequenceTagger
from flair.data import Sentence
import names
import spacy
from tqdm import tqdm
import numpy as np

root_dir = '.'
# meta_dir = os.path.join(root_dir, 'dataset_meta')
meta_dir = '/work/vig/qianrul/tofindwaldo/dataset_meta'
splits_dir = os.path.join(meta_dir, 'splits')
whos_waldo_dir = os.path.join(root_dir, 'whos_waldo')

nlp = spacy.load("en_core_web_trf", disable=[])
# doc = nlp(sample)
# for token in doc: 
#     print(token.text, token.pos_)
# exit(0)

def add_period(caption):
    if caption[-1] != '.':
        caption += '.'
    return caption

def get_real_name_caption(id):
    folder_path = os.path.join(whos_waldo_dir, id)
    corefs = read_json(os.path.join(folder_path, 'coreferences.json'))
    with open(os.path.join(folder_path, 'caption.txt'), 'r') as f:
        caption = f.readlines()[0].strip('\n')
    
    if len(corefs) == 0:
        return add_period(caption)

    generated_names = popular_names[:len(corefs)]
    spans = []
    spans_name = []
    for coref, generated_name in zip(corefs, generated_names):
        for span in coref:
            spans.append(span)
            spans_name.append(generated_name)
    
    inds = np.argsort(np.array(spans), axis=0)[:, 0]
    offset = 0
    for ind in inds:
        caption = caption[:spans[ind][0]+offset] + spans_name[ind] + caption[spans[ind][1]+offset:]
        offset += len(spans_name[ind]) - 6
    return add_period(caption)

def read_txt(txt_file):
    items = []
    with open(txt_file, 'r') as f:
        for item in f.readlines():
            items.append(item.strip())
    return items

def write_txt(txt_file, items):
    with open(txt_file, 'w') as fp:
        for item in items:
            fp.write("%s\n" % item)

def read_json(json_path):
    f = json.load(open(json_path, 'r', encoding='utf-8'))
    return f

def is_verb(token):
    return token.pos_ == "VERB" or token.pos_ == "AUX"

# # Get id of train set.
# train_ids = read_txt(os.path.join(splits_dir, 'train.txt'))
# print(len(train_ids))

# # Get verbs set from train set.
# train_verbs = set()
# for train_id in tqdm(train_ids):
#     caption = get_real_name_caption(train_id)
#     doc = nlp(caption)
#     for token in doc: 
#         if is_verb(token):
#             train_verbs.add(token.lemma_)
# print(len(train_verbs))
# print(train_verbs)
# train_verbs = list(train_verbs)
# write_txt(os.path.join(meta_dir, 'train_verbs.txt'), train_verbs)
# exit(0)

train_verbs = read_txt(os.path.join(meta_dir, 'train_verbs.txt'))
# Get id of test set.
test_ids = read_txt(os.path.join(splits_dir, 'test.txt'))
print(len(test_ids))
print(len(train_verbs))

# Get test image id whose caption has unseen verbs.
unseen_ids = []
no_verb_ids = []
for test_id in tqdm(test_ids):
    caption = get_real_name_caption(test_id)
    doc = nlp(caption)
    is_unseen = True
    has_meeting = 'meeting' in caption or 'Meeting' in caption
    has_verb = has_meeting
    for token in doc: 
        if is_verb(token):
            has_verb = True
        # if is_verb(token) and token.lemma_ not in train_verbs:
        #     is_unseen = True
        #     break
        if is_verb(token) and token.lemma_ in train_verbs:
            is_unseen = False
            break
    if not has_verb:
        print(test_id)
        print(caption)
        no_verb_ids.append(test_id)
    if is_unseen and has_verb and not has_meeting:
        unseen_ids.append(test_id)
print(len(unseen_ids))
print(unseen_ids)
print(len(no_verb_ids))
seen_ids = [id for id in test_ids if (id not in unseen_ids and id not in no_verb_ids)]
write_txt(os.path.join(meta_dir, 'all_unseen_img_ids.txt'), unseen_ids)
write_txt(os.path.join(meta_dir, 'all_seen_img_ids.txt'), seen_ids)
write_txt(os.path.join(meta_dir, 'no_verb_img_ids.txt'), no_verb_ids)
exit(0)
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
tagger = SequenceTagger.load('pos-fast')
chunker = SequenceTagger.load('chunk-fast')

interactive_imgs = []

for i in range(271747):
    folder = '%06d' % i
    folder_path = os.path.join(whos_waldo_dir, folder)
    corefs = read_json(os.path.join(folder_path, 'coreferences.json'))
    detections = read_json(os.path.join(folder_path, 'detections.json'))
    with open(os.path.join(folder_path, 'caption.txt'), 'r') as f:
        caption = f.readlines()[0].strip('\n')
    if len(detections) < 2: continue
    if len(corefs) < 2: continue
    has_verb = False

    sentences = sent_detector.tokenize(caption)
    sentences = [Sentence(s, use_tokenizer=True) for s in sentences]
    tagger.predict(sentences)
    chunker.predict(sentences)

    for s in sentences:
        VERB_spans = list(filter(lambda sp: sp.tag[:2] == 'VB', s.get_spans('pos')))
        if len(VERB_spans) > 0:
            has_verb = True
            break
    if not has_verb: continue
    interactive_imgs.append(folder)

with open(os.path.join(meta_dir, 'interactive_img_ids.txt'), 'w') as fp:
    for item in interactive_imgs:
        fp.write("%s\n" % item)