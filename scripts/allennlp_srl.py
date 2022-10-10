from custom_predictor import Predictor
from flair.data import Sentence
from flair.models import SequenceTagger
import os
import json
from tqdm import tqdm
import spacy
import numpy as np

tagger = SequenceTagger.load('pos')
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz", 'custom_semantic_role_labeling')
nlp = spacy.load('en_core_web_sm')

input_dir = '/work/vig/qianrul/whos-waldo/whos_waldo'
output_dir = '/work/vig/qianrul/whos-waldo/srl/bert-srl-flair-full'
meta_dir = '/work/vig/qianrul/tofindwaldo/dataset_meta'
splits_dir = os.path.join(meta_dir, 'splits')
popular_name_file = '/work/vig/qianrul/whos-waldo/scripts/popular_first_name_80.txt'
military_ranks_file = '/work/vig/qianrul/whos-waldo/scripts/military_ranks.txt'

np.random.seed(0)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

def read_txt(txt_file):
    items = []
    with open(txt_file, 'r') as f:
        for item in f.readlines():
            items.append(item.strip())
    return items

def read_json(json_path):
    f = json.load(open(json_path, 'r', encoding='utf-8'))
    return f

def find_all_spans(str, sub):
    spans = []
    start = 0
    while True:
        start = str.find(sub, start)
        if start == -1:
            break
        spans.append([start, start+len(sub)])
        start += len(sub)
    return spans

def replace_name(caption, corefs):
    # Get list of popular names for replacement.
    selected_names = np.random.choice(popular_names, len(corefs), replace=False)
    # Get mask of selected names and replace [NAME] with coref_id.
    name_masks = []
    for coref_id, coref in enumerate(corefs):
        org_name_spans = find_all_spans(caption, selected_names[coref_id])
        all_span = np.array(coref + org_name_spans)
        sorted_index = np.argsort(all_span[:, 0])
        coref_mask = np.ones(len(coref))
        org_mask = np.zeros(len(org_name_spans))
        name_mask = np.concatenate([coref_mask, org_mask])[sorted_index]
        name_masks.append(name_mask)
        for span in coref:
            caption = caption[:span[0]] + f'[{str(coref_id).zfill(4)}]' + caption[span[1]:]
    # Get replaced caption.
    for coref_id in range(len(corefs)):
        caption = caption.replace(f'[{str(coref_id).zfill(4)}]', selected_names[coref_id])
    return caption, selected_names, name_masks

def replace_abbr(caption, military_ranks):
    for abbr, full in military_ranks.items():
        caption = caption.replace(abbr, full)
    return caption

def get_name_index(srl_results, selected_names, name_masks, corefs, id):
    names_index = []
    for name_idx, selected_name in enumerate(selected_names):
        name_cnt = 0
        name_index = []
        for i, srl_result in enumerate(srl_results):
            for j, word in enumerate(srl_result['words']):
                if word.find(selected_name) != -1:
                    for _ in range(word.count(selected_name)):
                        if name_masks[name_idx][name_cnt] != 0:
                            name_index.append([i, j])
                        name_cnt += 1
        coref = corefs[name_idx]
        try:
            assert(len(coref) == len(name_index))
        except AssertionError:
            print(corefs)
            print(name_index)
            print(selected_name)
            print(id)
            print(srl_results)
            print(name_masks)
        sorted_index = np.array(coref)[:, 0].argsort()
        ranks = np.empty_like(sorted_index)
        ranks[sorted_index] = np.arange(len(coref))
        names_index.append(np.array(name_index)[ranks].tolist())
    return names_index

popular_names = []
popular_names_lines = read_txt(popular_name_file)
for line in popular_names_lines:
    content = line.split()
    popular_names += [content[1], content[3]]
popular_names = np.array(popular_names)

military_ranks = {}
military_ranks_lines = read_txt(military_ranks_file)
for line in military_ranks_lines:
    content = line.split()
    military_ranks[content[0]] = content[1]

# last_id = '208816'
# begin = False
# ['train', 'val', 'test']
for split in ['train', 'val', 'test']:
    ids = read_txt(os.path.join(splits_dir, f'{split}.txt'))
    for id in tqdm(ids):
        caption_file = os.path.join(input_dir, id, 'caption.txt')
        output_path = os.path.join(output_dir, split)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        output_file = os.path.join(output_path, f'{id}.json')

        # Load data.
        corefs = read_json(os.path.join(input_dir, id, 'coreferences.json'))
        with open(caption_file, 'r') as f:
            caption = f.readlines()[0].strip('\n')

        # Preprocess captions.
        caption, selected_names, name_masks = replace_name(caption, corefs)
        # if id == last_id:
        #     begin = True
        # if not begin:
        #     continue
        caption = caption.rstrip()
        if caption[-1] != '.':
            caption += '.'
        caption = replace_abbr(caption, military_ranks)
        sentences = [sent.text.strip() for sent in nlp(caption).sents]

        # Get srl results.
        results = {}
        srl_results = []
        for sentence in sentences:
            flairSentence = Sentence(sentence)
            tagger.predict(flairSentence)
            labels = [label.value for label in flairSentence.labels]
            tokens = [token.text for token in flairSentence]
            result = predictor.predict_tokenized(tokenized_sentence=tokens, labels=labels)
            srl_results.append(result) 
        results['srl_results'] = srl_results
        results['names_index'] = get_name_index(srl_results, selected_names, name_masks, corefs, id)
        results['selected_names'] = selected_names.tolist()
        with open(output_file, "w") as f:
            json.dump(results, f)
        
