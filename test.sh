#!/usr/bin/env bash

set -x
source /work/vig/qianrul/whosWaldo.sh

txt_db='0806-new'
img_db='0806-R101-k36-new'
model_dir='finetune-0806-new'
ckpt='50000'
split='test'
eval_output_name='1209-R101-k36-new-50000-all'
batch_size=1024
n_workers=4
visibility='all'

CUDA_VISIBLE_DEVICES=0 python infer.py --txt-db=${txt_db} \
        --img-db=${img_db} \
        --model-dir=${model_dir} \
        --ckpt=${ckpt} \
        --split=${split} \
        --eval-output-name=${eval_output_name} \
        --batch_size=${batch_size} \
        --n_workers=${n_workers} \
        --pin_mem \
        --visibility=${visibility} \
        