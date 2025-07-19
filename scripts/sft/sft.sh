#! /bin/bash

accelerate launch \
    --config-file ./scripts/sft/sft.yaml \
    sft.py \
    --config-file-path ./configs/sft.yaml
