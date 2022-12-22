#!/usr/bin/env bash

set -x
source /work/vig/qianrul/whosWaldo.sh

txt_db='clip'
img_db='clip'
model_dir='finetune-1210-new-clip-person'
ckpt='3500'
split='test'
eval_output_name='test-clip-3500'
batch_size=1024
n_workers=4
use_clip=true
visibility='all'

CUDA_VISIBLE_DEVICES=0 python infer.py --txt-db=${txt_db} \
        --img-db=${img_db} \
        --model-dir=${model_dir} \
        --ckpt=${ckpt} \
        --split=${split} \
        --eval-output-name=${eval_output_name} \
        --batch_size=${batch_size} \
        --n_workers=${n_workers} \
        --visibility=${visibility} \
        --use_clip=${use_clip} \
        --pin_mem