import numpy as np
import clip
import torch
import torchvision.transforms.functional as F
import os
from PIL import Image
from tqdm import tqdm

old_visual_dir = './whos-waldo-features-R101-k36'
new_visual_dir = './whos-waldo-features-clip'
data_path = '/work/vig/qianrul/whos-waldo/whos_waldo'
clip_model_name = 'ViT-B/32'
device = 'cuda'

def read_txt(f):
    return [n.rstrip('\n') for n in f.readlines()]

def get_clip_features():
    clip_model, clip_preprocess = clip.load(clip_model_name, device=device)
    image_encoder = clip_model.encode_image
    # textual_encoder = clip_model.encode_text
    ids = os.listdir(old_visual_dir)
    for id in tqdm(ids):
        data = dict(np.load(f'{old_visual_dir}/{id}/{id}.npz', allow_pickle=True))
        boxes = torch.as_tensor(data['bbox'], dtype=torch.float32).reshape(-1, 4)
        img = Image.open(f'{data_path}/{id}/image.jpg').convert('RGB')
        # with open(f'{data_path}/{id}/caption.txt', 'r') as f:
        #     text = read_txt(f)[0]
        clip_box_inputs = []
        for box in boxes:
            box_img = F.crop(img=img, top=box[1].item(), left=box[0].item(), height=box[3].item()-box[1].item(), width=box[2].item()-box[0].item())
            clip_box_inputs.append(clip_preprocess(box_img))
        clip_box_inputs = torch.stack(clip_box_inputs, dim=0).cuda()
        # text_tokens = clip.tokenize(text, truncate=True).cuda()    
        with torch.no_grad():
            image_features = image_encoder(clip_box_inputs).cpu().numpy()
            # textual_features = textual_encoder(text_tokens, use_word_features=True).squeeze(0).cpu().numpy()
        # print(text_tokens.argmax(dim=-1).item())
        # print(image_features)
        # print(textual_features.shape)
        assert(data['x'].shape[0] == image_features.shape[0])
        data['x'] = image_features
        # data['txt'] = textual_features
        # data['eot'] = text_tokens.argmax(dim=-1).item()
        if not os.path.exists(f'{new_visual_dir}/{id}'):
            os.mkdir(f'{new_visual_dir}/{id}')
        np.savez(f'{new_visual_dir}/{id}/{id}.npz', **data)

get_clip_features()