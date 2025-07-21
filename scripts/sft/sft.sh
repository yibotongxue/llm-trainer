#! /bin/bash

accelerate launch --config-file ./scripts/sft/sft.yaml ./scripts/sft/sft.py --config-file-path ./configs/sft.yaml
