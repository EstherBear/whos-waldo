import json
import os
from tqdm import tqdm

data_path = '/work/vig/qianrul/whos-waldo/whos_waldo'
splits_path = '/work/vig/qianrul/tofindwaldo/dataset_meta/splits'
srl_path = '/work/vig/qianrul/whos-waldo/srl-triplet/bert-srl-flair-full'
split = 'train'
min_pair_num = 2

def read_txt(txt_file: str) -> list[str]:
	items = []
	with open(txt_file, 'r') as f:
		for item in f.readlines():
			items.append(item.strip())
	return items

def write_txt(txt_file: str, items: list[str]):
	with open(txt_file, 'w') as f:
		for item in items:
			f.write(f'{item}\n')

def read_json(json_path: str):
	f = json.load(open(json_path, 'r', encoding='utf-8'))
	return f

img_ids = read_txt(os.path.join(splits_path, f'{split}.txt'))
print(len(img_ids))
filtered_img_ids = []
for img_id in tqdm(img_ids):
	gt = read_json(os.path.join(data_path, img_id, 'ground_truth.json'))
	if len(gt) < 2:
		continue
	srl_triplets = read_json(os.path.join(srl_path, split, f'{img_id}.json'))['triplets'] # [[[0], "met", [1]]]
	gt_pairs = set()
	for verb_triplets in srl_triplets:
		sub_corefs = verb_triplets[0]
		obj_corefs = verb_triplets[2]
		gt_sub_corefs = [str(sub_coref) for sub_coref in sub_corefs if str(sub_coref) in gt]
		gt_obj_corefs = [str(obj_coref) for obj_coref in obj_corefs if str(obj_coref) in gt]
		
		for gt_sub in gt_sub_corefs:
			for gt_obj in gt_obj_corefs:

				# We don't expect a person to be both subject and object.
				if gt_sub == gt_obj:
					continue
				gt_pairs.add((gt_sub, gt_obj))
				
		if len(gt_pairs) >= min_pair_num:
			filtered_img_ids.append(img_id)
			break

print(filtered_img_ids)
print(len(filtered_img_ids))
write_txt(os.path.join(splits_path, f'{split}_triplet{min_pair_num}.txt'), filtered_img_ids)
