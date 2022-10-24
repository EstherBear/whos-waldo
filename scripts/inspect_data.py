import json
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from PIL import Image
from nltk import word_tokenize

data_dir = '/work/vig/qianrul/whos-waldo/whos_waldo'
split = 'test'
srl_dir = f'/work/vig/qianrul/whos-waldo/srl-triplet/bert-srl-flair-full/{split}'
splits_dir = '/work/vig/qianrul/tofindwaldo/dataset_meta/splits'

id = '018539'

def read_txt(txt_file):
    items = []
    with open(txt_file, 'r') as f:
        for item in f.readlines():
            items.append(item.strip())
    return items

def read_json(json_path):
    f = json.load(open(json_path, 'r', encoding='utf-8'))
    return f

output_path = f'/work/vig/qianrul/whos-waldo/scripts/tmp/srl/bert-srl-flair-full/{split}'
if not os.path.exists(output_path):
    os.mkdir(output_path)
colors = list(mcolors.CSS4_COLORS.keys())
caption_file = os.path.join(data_dir, id, 'caption.txt')
with open(caption_file, 'r') as f:
	caption = f.readlines()[0].strip('\n')
corefs = read_json(os.path.join(data_dir, id, 'coreferences.json'))

gt = read_json(os.path.join(data_dir, id, 'ground_truth.json'))

for coref_id, coref in enumerate(corefs):
    if str(coref_id) in gt:
        bbox_id = f'[{str(gt[str(coref_id)]).zfill(4)}]'
    else:
        bbox_id = f'[xxxx]'
    for interval in coref:
        caption = caption[:interval[0]] + bbox_id + caption[interval[1]:]

srl = read_json(os.path.join(srl_dir, f'{id}.json'))

detections = read_json(os.path.join(data_dir, id, 'detections.json'))
bboxs = [detection['bbox'] for detection in detections]
bbox_num = len(bboxs)

img_path = os.path.join(data_dir, id, 'image.jpg')
bbox_color = np.random.randint(low=0, high=len(colors), size=bbox_num)
plt.figure()
fig, ax = plt.subplots(1)
img = Image.open(img_path)
width, height = img.width, img.height
ax.imshow(img)

for i, bbox in enumerate(bboxs):
    x1, y1, x2, y2 = bbox[:4]
    x1 *= width
    x2 *= width
    y1 *= height
    y2 *= height
    # print(x1, y1, x2, y2)
    w = x2 - x1
    h = y2 - y1
    color = bbox_color[i]
    rects = patches.Rectangle((x1,y1), w, h, linewidth=2, edgecolor = colors[color], facecolor='none')
    plt.text(x1, y1, str(i), color='lime', verticalalignment ='top')
    ax.add_patch(rects)
plt.axis('off')
fig.savefig(f'{output_path}/{id}.jpg', bbox_inches='tight', pad_inches= 0.0)
plt.close('all')
print(caption)
print(corefs)
print(srl)
print(gt)
# # Max corefs number.
# meta_dir = '/work/vig/qianrul/tofindwaldo/dataset_meta'
# splits_dir = os.path.join(meta_dir, 'splits')
# data_dir = '/work/vig/qianrul/whos-waldo/whos_waldo'

# max_corefs_num = 0
# for split in ['train', 'val', 'test']:
#     ids = read_txt(os.path.join(splits_dir, f'{split}.txt'))
#     for id in tqdm(ids):
#         corefs = read_json(os.path.join(data_dir, id, 'coreferences.json'))
#         max_corefs_num = max(max_corefs_num, len(corefs))

# print(max_corefs_num)

# # Max srl argument.
# meta_dir = '/work/vig/qianrul/tofindwaldo/dataset_meta'
# splits_dir = os.path.join(meta_dir, 'splits')
# data_dir = '/work/vig/qianrul/whos-waldo/whos_waldo'

# max_corefs_num = 0
# for split in ['train', 'val', 'test']:
#     ids = read_txt(os.path.join(splits_dir, f'{split}.txt'))
#     for id in tqdm(ids):
#         corefs = read_json(os.path.join(data_dir, id, 'coreferences.json'))
#         max_corefs_num = max(max_corefs_num, len(corefs))

# # Who's Waldo vocabulary.
# vocabulary = dict()
# for split in ['train', 'val', 'test']:
#     ids = read_txt(os.path.join(splits_dir, f'{split}.txt'))
#     for id in tqdm(ids):
#         caption_file = os.path.join(data_dir, id, 'caption.txt')
#         with open(caption_file, 'r') as f:
#             caption = f.readlines()[0].strip('\n')
#             words = word_tokenize(caption)
#             for word in words:
#                 if word not in vocabulary:
#                     vocabulary[word] = 1
#                 else:
#                     vocabulary[word] += 1

# output_file = '/work/vig/qianrul/whos-waldo/scripts/vocabulary.txt'
# with open(output_file, "w") as f:
#     for word in sorted(vocabulary.keys()):
#         f.write(f'{word}\t{vocabulary[word]}\n')