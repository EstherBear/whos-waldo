import os
from tqdm import tqdm
import json
import re
import numpy as np

meta_dir = '/work/vig/qianrul/tofindwaldo/dataset_meta'
splits_dir = os.path.join(meta_dir, 'splits')
data_dir = '/work/vig/qianrul/whos-waldo/whos_waldo'

def read_txt(txt_file):
    items = []
    with open(txt_file, 'r') as f:
        for item in f.readlines():
            items.append(item.strip())
    return items

def read_json(json_path):
    f = json.load(open(json_path, 'r', encoding='utf-8'))
    return f

split = 'val'
srl_dir = f'/work/vig/qianrul/whos-waldo/srl/bert-srl-flair-full/{split}'
output_dir = '/work/vig/qianrul/whos-waldo/srl/bert-srl-flair-full'
ids = read_txt(os.path.join(splits_dir, f'{split}.txt'))
total_corefs = 0
untag_corefs = 0
untag_ids = []
for id in tqdm(ids):
    # Load data.
    caption_file = os.path.join(data_dir, id, 'caption.txt')
    with open(caption_file, 'r') as f:
        org_caption = f.readlines()[0].strip('\n')
    corefs = read_json(os.path.join(data_dir, id, 'coreferences.json'))
    srl = read_json(os.path.join(srl_dir, f'{id}.json'))

    # Preprocess caption.
    caption = org_caption.rstrip()
    if caption[-1] != '.':
        caption += '.'
    # print(caption)

    
    # Judge if a [NAME] has tag.
    for coref_idx, coref in enumerate(corefs):
        total_corefs += 1
        has_tag = False
        for span_idx, span in enumerate(coref):
            i, j = srl['names_index'][coref_idx][span_idx]
            verbs = srl['srl_results'][i]['verbs']
            for verb in verbs:
                if verb['tags'][j] != 'O':
                    has_tag = True
                    break
            if has_tag:
                break
        if not has_tag:
            untag_corefs += 1
            untag_ids.append({'id': id, 'coref_idx': coref_idx})
            # print(coref)
print(total_corefs)
print(untag_corefs)
print(untag_corefs / total_corefs)

output = {}
output['total_corefs'] = total_corefs
output['untag_corefs'] = untag_corefs
output['untag_ration'] = untag_corefs / total_corefs
output['untag_ids'] = untag_ids
with open(f'{output_dir}/{split}_untag.json', "w") as f:
    json.dump(output, f)




