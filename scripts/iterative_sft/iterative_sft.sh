#! /bin/bash

export QWEN_API_KEY=your_api_key_here
export DEEPSEEK_API_KEY=your_api_key_here

accelerate launch --config-file ./scripts/iterative_sft/iterative_sft.yaml ./scripts/iterative_sft/iterative_sft.py --config-file-path ./configs/iterative_sft.yaml
