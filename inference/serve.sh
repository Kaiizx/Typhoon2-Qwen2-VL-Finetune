#!/bin/bash

which python3

vllm serve "/home/atikan/ss5/Qwen2-VL-Finetune/merged_weight/typhoon2-ft-3e" --task generate --allowed-local-media-path /home/atikan/ss5/Qwen2-VL-Finetune/inference/test \
  --trust-remote-code --max-model-len 4096 --limit-mm-per-prompt image=1 