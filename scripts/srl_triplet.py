import os
import json
from tqdm import tqdm
import numpy as np
from word_forms.word_forms import get_word_forms
from word_forms.lemmatizer import lemmatize

input_dir = '/work/vig/qianrul/whos-waldo/srl/bert-srl-flair-full'
output_dir = '/work/vig/qianrul/whos-waldo/srl-triplet/bert-srl-flair-full'
splits_dir = '/work/vig/qianrul/tofindwaldo/dataset_meta/splits'

def read_txt(txt_file):
    items = []
    with open(txt_file, 'r') as f:
        for item in f.readlines():
            items.append(item.strip())
    return items

def read_json(json_path):
    f = json.load(open(json_path, 'r', encoding='utf-8'))
    return f

def get_ing(verb: str) -> str:
    verb_forms = get_word_forms(verb)['v']
    for verb_form in verb_forms:
        if verb_form[-3:] == 'ing':
            return verb_form
    return verb

def get_ed(verb: str) -> str:
    verb_forms = get_word_forms(verb)['v']
    for verb_form in verb_forms:
        if verb_form[-2:] == 'ed':
            return verb_form
    return verb

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# triplet: ([sub_id], verb, [obj_id])
for split in ['test', 'val', 'train']:
    ids = read_txt(os.path.join(splits_dir, f'{split}.txt'))
    input_split_dir = os.path.join(input_dir, split)
    output_path = os.path.join(output_dir, split)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for id in tqdm(ids):
        srl = read_json(os.path.join(input_split_dir, f'{id}.json'))
        triplets = []
        names_index = srl['names_index']
        for result_id, srl_result in enumerate(srl['srl_results']):
            for verb_result in srl_result['verbs']:
                verb = verb_result['verb']
                sub = []
                obj = []
                tags = verb_result['tags']
                for coref_id, name_index in enumerate(names_index):
                    for coref_index in name_index:
                        if coref_index[0] == result_id and 'ARG' in tags[coref_index[1]]:
                            if 'ARG0' in tags[coref_index[1]]:
                                sub.append(coref_id)
                            else:
                                obj.append(coref_id)
                if len(sub) == 0 and len(obj) == 0:
                    continue
                if len(sub) == 0 and len(obj) > 0:
                    triplets.append(([*set(sub)], get_ed(verb), [*set(obj)]))
                else:    
                    triplets.append(([*set(sub)], get_ing(verb), [*set(obj)]))
        srl['triplets'] = triplets
        output_file = os.path.join(output_path, f'{id}.json')
        with open(output_file, "w") as f:
            json.dump(srl, f)

                            

