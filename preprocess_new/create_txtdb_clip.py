"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Process annotations into LMDB
"""
import argparse
import json
import os, io, shutil, sys

sys.path.append('.')
from os.path import exists
from cytoolz import curry
from pytorch_pretrained_bert import BertTokenizer
from data.data import open_lmdb
import re

import clip
import torch

annotation_dir = './storage/annotations'
txt_db_dir = './storage/txt_db'


def tokenize(tokenizer, text):
    ids = []
    for word in text.strip().split():
        ws = tokenizer.tokenize(word)
        if not ws: continue # some special char
        ids.extend(tokenizer.convert_tokens_to_ids(ws))
    return ids


def name_token_to_lower(caption):
    return caption.replace('[NAME]', '[name]')

def name_token_to_start(caption):
    caption = caption.replace('[name]', ' [name] ')
    return caption.replace('[name]', '<|startoftext|>')

def name_token_to_person(caption):
    caption = caption.replace('[name]', ' [name] ')
    name_token_pos = [m.start() for m in re.finditer('\\[name\\]', caption)]
    person_pos = [m.start() for m in re.finditer(' person | person,|-person,| person\.| person\'|-person | Councilperson |a drowning person| person:| staffperson |\'Person ', caption, flags=re.IGNORECASE)]
    # print(caption)
    # print(name_token_pos)
    # print(person_pos)
    # exit(0)
    total_len = len(name_token_pos) + len(person_pos)
    mask = [0 for _ in range(total_len)]
    p1 = 0
    p2 = 0
    pmask = 0
    while p1 < len(name_token_pos) and p2 < len(person_pos):
        if name_token_pos[p1] < person_pos[p2]:
            p1 += 1
        else:
            mask[p1+p2] = 1
            p2 += 1
    if p2 < len(person_pos):
        for i in range(p1+p2, total_len):
            mask[i] = 1
    return caption.replace('[name]', 'person'), mask    

def replace_name_token_cased(tokens):
    """
    :param tokens: tokens output by the cased BertTokenizer
    :return: tokens with the sequence 164('['), 1271('name'), 166(']') replaced by 104 ('[NAME]')
    """
    while 1271 in tokens:
        i = tokens.index(1271)
        if i - 1 >= 0 and i + 1 < len(tokens) and tokens[i - 1] == 164 and tokens[i + 1] == 166:
            tokens[i - 1] = 104
            del tokens[i + 1]
            del tokens[i]
        else:
            tokens[i] = 105
    for i in range(len(tokens)):
        if tokens[i] == 105: tokens[i] = 1271
    return tokens

def replace_name_token_cased_to_person(tokens):
    """
    :param tokens: tokens output by the cased BertTokenizer
    :return: tokens with the sequence 314('['), 1981('name'), 316(']') replaced by 2533 ('person')
    """
    while 1981 in tokens:
        i = tokens.index(1981)
        if i - 1 >= 0 and i + 1 < len(tokens) and tokens[i - 1] == 314 and tokens[i + 1] == 316:
            tokens[i - 1] = 2533
            del tokens[i + 1]
            del tokens[i]
            tokens.append(0)
            tokens.append(0)
        
        else:
            tokens[i] = 49408
    for i in range(len(tokens)):
        if tokens[i] == 49408: tokens[i] = 1981
    return tokens


def process_people(opt, db, tokenizer):
    clip_model_name = 'ViT-B/32'
    device = 'cuda'
    clip_model, clip_preprocess = clip.load(clip_model_name, device=device)
    textual_encoder = clip_model.encode_text

    annotation_json_path = os.path.join(annotation_dir, opt.ann, opt.split + '.json')
    examples = json.load(io.open(annotation_json_path, 'r', encoding='utf-8'))
    print(f'loaded {len(examples)} examples for {opt.split}...')

    id2len = {}
    id2category = {}
    exceed = 0
    for example in examples:
        id_ = example['id']
        caption_lower = name_token_to_lower(example['caption'])        
        tokens = tokenize(tokenizer, caption_lower)
        input_ids = replace_name_token_cased(tokens)  # all the [name] tokens map to 104
        try:
            clip_caption, mask = name_token_to_person(caption_lower)
            clip_tokens = clip.tokenize(clip_caption)
        except RuntimeError:
            print(f'{id_}') 
            exceed += 1
            continue      
        # print(clip_tokens.squeeze(0).tolist()) 
        # clip_input_ids = replace_name_token_cased_to_sot(clip_tokens.squeeze(0).tolist())
        # clip_tokens = torch.tensor(clip_input_ids).unsqueeze(0) 
        clip_tokens_len = clip_tokens.argmax(dim=-1).item() + 1
        clip_input_ids = clip_tokens.squeeze(0).tolist()
        # print(clip_input_ids)
        # print(clip.decode(clip_input_ids))
        # print(clip_tokens_len)
        with torch.no_grad():
            textual_features = textual_encoder(clip_tokens.cuda(), use_word_features=True).squeeze(0).cpu().numpy()
        # print(input_ids)
        # print(len(input_ids))
        # print(clip_input_ids)
        # print(len(clip_input_ids))
        # print(caption_lower)
        # print(clip.decode([58]))
        # print(clip.decode([49408]))
        # print(textual_features.shape)
        # print(clip.decode([314]))
        # print(clip.decode([1981]))
        # print(clip.decode([316]))
        # print(clip.decode([34067]))
        # print(caption_lower)
        # exit(0)
        name_pos = [i for i in range(len(input_ids)) if input_ids[i] == 104]
        mask_idx = 0
        clip_name_pos = []
        for i in range(1, len(clip_input_ids)):
            if clip_input_ids[i] == 2533:
                if mask_idx >= len(mask):
                    print(f'{id_}: {caption_lower}') 
                    print(clip_input_ids)
                    print(clip_caption)
                    print(mask)        
                if mask[mask_idx] != 1:
                    clip_name_pos.append(i - 1)
                mask_idx += 1
        # clip_name_pos = [i - 1 for i in range(1, len(clip_input_ids)) if clip_input_ids[i] == 2533]
        assert len(name_pos) == len(example['corefs'])
        if len(clip_name_pos) != len(example['corefs']):
            print(f'{id_}: {caption_lower}') 
            print(clip_input_ids)
            print(clip_caption)
            print(mask)
            print(clip.decode([1]))
            print(clip.decode([27]))
            print(clip.decode([347]))
        assert len(clip_name_pos) == len(example['corefs'])
        iden2token_pos = {}
        clip_iden2token_pos = {}
        for i in range(len(name_pos)):
            iden = example['corefs'][i][1]
            if not iden in iden2token_pos.keys():
                iden2token_pos[iden] = [name_pos[i]]
            else:
                iden2token_pos[iden].append(name_pos[i])
        for i in range(len(clip_name_pos)):
            iden = example['corefs'][i][1]
            if not iden in clip_iden2token_pos.keys():
                clip_iden2token_pos[iden] = [clip_name_pos[i]]
            else:
                clip_iden2token_pos[iden].append(clip_name_pos[i])

        gt_rev = {v: k for k, v in example['gt'].items()}
        example['gt_rev'] = gt_rev
        example['iden2token_pos'] = clip_iden2token_pos
        # example['clip_iden2token_pos'] = clip_iden2token_pos
        example['input_ids'] = clip_input_ids
        example['txt_feat'] = textual_features
        # example['clip_input_ids'] = clip_input_ids
        db[id_] = example
        # id2len[id_] = len(input_ids)
        id2len[id_] = clip_tokens_len - 2
        id2category[id_] = example['category']
    print(f'database length for {opt.split} is {len(examples) - exceed}')
    return id2len, id2category


def main(opts):
    output_dir = os.path.join(txt_db_dir, opts.output, opts.split)
    if exists(output_dir):
        shutil.rmtree(output_dir)
        print("Removed existing DB at " + output_dir)

    os.makedirs(output_dir)
    meta = vars(opts)
    meta['tokenizer'] = opts.tokenizer

    tokenizer = BertTokenizer(opts.vocab, do_lower_case=False)
    meta['UNK'] = tokenizer.convert_tokens_to_ids(['[UNK]'])[0]
    meta['CLS'] = tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
    meta['SEP'] = tokenizer.convert_tokens_to_ids(['[SEP]'])[0]
    meta['MASK'] = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
    meta['NAME'] = tokenizer.convert_tokens_to_ids(['[NAME]'])[0]
    meta['v_range'] = (tokenizer.convert_tokens_to_ids('!')[0],
                       len(tokenizer.vocab))
    with open(f'{output_dir}/meta.json', 'w') as f:
        json.dump(vars(opts), f, indent=4)

    open_db = curry(open_lmdb, output_dir, readonly=False)
    with open_db() as db:
        id2len, id2category = process_people(opts, db, tokenizer)
    with open(f'{output_dir}/id2len.json', 'w') as f:
        json.dump(id2len, f)
    with open(f'{output_dir}/id2category.json', 'w') as f:
        json.dump(id2category, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', required=True,
                        help='which split')
    parser.add_argument('--ann', required=True,
                        help='name of directory containing annotations')
    parser.add_argument('--output', required=True,
                        help='output dir of DB')
    parser.add_argument('--vocab', default='./dataset_meta/vocab-cased-with-name.txt',
                        help='vocabulary for tokenizer')
    parser.add_argument('--tokenizer', default='bert-base-cased',
                        help='which BERT tokenizer to used')

    args = parser.parse_args()
    main(args)