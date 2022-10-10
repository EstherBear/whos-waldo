import os
import json
from tqdm import tqdm
import spacy
import numpy as np

nlp = spacy.load('en_core_web_sm')

input_dir = '/work/vig/qianrul/whos-waldo/whos_waldo'
meta_dir = '/work/vig/qianrul/tofindwaldo/dataset_meta'
splits_dir = os.path.join(meta_dir, 'splits')
output_dir = '/work/vig/qianrul/software/tanl/data/WhosWaldo'
popular_name_file = '/work/vig/qianrul/whos-waldo/scripts/popular_first_name_80.txt'

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

def is_verb(token):
    return token.pos_ == "VERB" or token.pos_ == "AUX"

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
        name_mask = np.concatenate([coref_mask, org_mask])[sorted_index].tolist()
        name_masks.append(name_mask)
        for span in coref:
            caption = caption[:span[0]] + f'[{str(coref_id).zfill(4)}]' + caption[span[1]:]
    # Get replaced caption.
    for coref_id in range(len(corefs)):
        caption = caption.replace(f'[{str(coref_id).zfill(4)}]', selected_names[coref_id])
    return caption, selected_names, name_masks

popular_names = []
popular_names_lines = read_txt(popular_name_file)
for line in popular_names_lines:
    content = line.split()
    popular_names += [content[1], content[3]]
popular_names = np.array(popular_names)

# ['train', 'val', 'test']
for split in ['test']:
    ids = read_txt(os.path.join(splits_dir, f'{split}.txt'))
    output_path = os.path.join(output_dir, split)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    captions_f = open(os.path.join(output_path, 'captions.txt'), "w")
    for id in tqdm(ids):
        caption_file = os.path.join(input_dir, id, 'caption.txt')
        output_file = os.path.join(output_path, f'{id}.json')

        # Load data.
        corefs = read_json(os.path.join(input_dir, id, 'coreferences.json'))
        with open(caption_file, 'r') as f:
            caption = f.readlines()[0].strip('\n')

        # Preprocess captions.
        caption, selected_names, name_masks = replace_name(caption, corefs)
        caption = caption.rstrip()
        if caption[-1] != '.':
            caption += '.'
        doc = nlp(caption)

        for sent in doc.sents:
            sentence_tags = []
            for token_id, token in enumerate(sent): 
                if is_verb(token):
                    tag = ['O' for _ in range(len(sent))]
                    tag[token_id] = 'B-V'
                    sentence_tags.append(tag)
            if len(sentence_tags) == 0:
                sentence_tags.append(['O' for _ in range(len(sent))])
                sentence_tags[0][-1] = 'B-V'
            for sentence_tag in sentence_tags:
                captions_f.write(f'{id} {sent.text.strip()} ||| {(" ").join(sentence_tag)}\n')

        # Get tanl input.
        results = {}
        results['name_masks'] = name_masks
        results['corefs'] = corefs
        results['id'] = id
        results['selected_names'] = selected_names.tolist()
        with open(output_file, "w") as f:
            json.dump(results, f)
