import sng_parser
import os, io
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from PIL import Image
import shutil
import spacy

split = 'train'
annotationJsonPath = "/work/vig/qianrul/whos-waldo/storage/annotations/0806-new/{}.json".format(split)
id2lenPath = "/work/vig/qianrul/whos-waldo/storage/txt_db/0806-new/{}/id2len.json".format(split)
examples = json.load(io.open(annotationJsonPath, 'r', encoding='utf-8'))
id2len = json.load(io.open(id2lenPath, 'r', encoding='utf-8'))
outputRoot = '/work/vig/qianrul/whos-waldo/scripts/tmp'
if not os.path.exists(outputRoot):
    os.mkdir(outputRoot)
nlp = spacy.load('en_core_web_sm')

def genGraph(example, outputPath):
    ID = example['id']
    caption = example['caption']
    caption = caption.replace('[NAME]', 'Jack')
    graph, doc = sng_parser.parse(caption, return_doc=True)
    # print(graph)
    # for chunk in doc.noun_chunks:
    #     print(chunk.text, chunk.root.text, chunk.root.dep_,
    #             chunk.root.head.text)
    sng_parser.tprint(graph)
    print(caption)
    dname = '{}_{}_{}'.format(id2len[ID], example['category'], ID)
    outputDir = '{}/{}'.format(outputPath, dname)
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)
    TSGFile = '{}/{}'.format(outputDir, "TSG.txt")
    with open(TSGFile, "w") as file: 
        sng_parser.tprint(graph, file=file)
        file.write(caption)

def genImage(example, outputPath):
    ID = example['id']
    imgNum = ID
    dname = '{}_{}_{}'.format(id2len[ID], example['category'], ID)
    outputDir = '{}/{}'.format(outputPath, dname)
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)
    root = '/work/vig/qianrul/whos-waldo/whos_waldo/{}'.format(imgNum)
    bboxPath = "{}/detections.json".format(root)
    imgPath = "{}/image.jpg".format(root)
    bboxs = []
    with open(bboxPath,'r') as load_f:
        load_dict = json.load(load_f)
        # print(load_dict)
    for detection in load_dict:
        bboxs.append(detection['bbox'])
        # print(len(detection['keypoints']))
    bboxs = np.array(bboxs)
    colors = list(mcolors.CSS4_COLORS.keys())
    bboxNum = bboxs.shape[0] 
    bboxColor = np.random.randint(low=0, high=len(colors), size=bboxNum)
    # print(bboxs)
    # print(bboxColor)
    plt.figure()
    fig, ax = plt.subplots(1)
    img = Image.open(imgPath)
    width, height = img.width, img.height
    ax.imshow(img)
    links = example['gt']
    print(links)
    plt.text(0, 0, str(links), color='red', verticalalignment ='top')
    for id, bbox in enumerate(bboxs):
        x1, y1, x2, y2 = bbox[:4]
        x1 *= width
        x2 *= width
        y1 *= height
        y2 *= height
        # print(x1, y1, x2, y2)
        w = x2 - x1
        h = y2 - y1
        color = bboxColor[id]
        rects = patches.Rectangle((x1,y1), w, h, linewidth=2, edgecolor = colors[color], facecolor='none')
        coref = None
        for c, d in links.items():
            if d == id:
                coref = c
                break
        print(coref)
        tokenIds = []
        for i in range(len(example['corefs'])):
            iden = str(example['corefs'][i][1])
            if iden == coref:
                tokenIds.append(str(i))
        print(colors[color], tokenIds)
        plt.text(x1, y1, ','.join(tokenIds), color='red', verticalalignment ='top')

        ax.add_patch(rects)


    plt.axis('off')
    fig.savefig('{}/{}.jpg'.format(outputDir, imgNum), bbox_inches='tight', pad_inches= 0.0)
    plt.close('all')

def genCaptions(example, outputPath, task):
    ID = example['id']
    caption = example['caption']
    caption = caption.replace('[NAME]', 'Jack')
    caption = caption.rstrip()
    if caption[-1] != '.':
        caption += '.'
    sentences = [i for i in nlp(caption).sents]
    outputDir = outputPath
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)
    Captions = '{}/{}{}'.format(outputDir, task, "Captions.txt")
    with open(Captions, "a") as file: 
        for sent in sentences:
            file.write('{}\t{}\n'.format(ID, sent))
# random
np.random.seed(0)
total = len(examples)
chosen = np.random.choice(total, 10, replace=False)
outputPath = '{}/random'.format(outputRoot)
if not os.path.exists(outputPath):
    os.mkdir(outputPath)
for idx in chosen:
    example = examples[idx]
    genImage(example, outputPath)
    genCaptions(example, outputPath, "Random")

# interactive
num = 10
cnt = 0
outputPath = '{}/interactive'.format(outputRoot)
if not os.path.exists(outputPath):
    os.mkdir(outputPath)
for example in examples:
    if example['category'] == 'interactive':
        cnt += 1
        genImage(example, outputPath)
        genCaptions(example, outputPath, "Interactive")
    if cnt == num:
        break

# len >= 50
num = 10
cnt = 0
outputPath = '/work/vig/qianrul/whos-waldo/scripts/tmp/long'
if not os.path.exists(outputPath):
    os.mkdir(outputPath)
for example in examples[::-1]:
    ID = example['id']
    if id2len[ID] >= 50:
        cnt += 1
        genImage(example, outputPath)
        genCaptions(example, outputPath, "LongerThan50")
    if cnt == num:
        break

