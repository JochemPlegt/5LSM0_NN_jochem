#!/bin/bash

BASE_ARGS="--data-dir ./data/cityscapes --batch-size 64 --epochs 100 --lr 0.001 --num-workers 10 --seed 42"

python3 train.py \
    $BASE_ARGS \
    --experiment-id "${EXPERIMENT:-unet-no-augment}" \
    ${AUGMENT_FLAGS:-}
