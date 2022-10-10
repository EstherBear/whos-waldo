import os
import json
from tqdm import tqdm
import numpy as np

input_dir = '/work/vig/qianrul/whos-waldo/bert-srl'
output_dir = '/work/vig/qianrul/whos-waldo/srl-triplet/bert-srl'
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

# triplet: ([sub_id], verb, [obj_id])

for split in ['train', 'val', 'test']:
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
                triplets.append(([*set(sub)], verb, [*set(obj)]))
        srl['triplets'] = triplets
        output_file = os.path.join(output_path, f'{id}.json')
        with open(output_file, "w") as f:
            json.dump(srl, f)

                            

