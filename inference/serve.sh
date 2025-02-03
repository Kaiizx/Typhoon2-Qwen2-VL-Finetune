#!/bin/bash

which python3

vllm serve "merged/model-path" --task generate --allowed-local-media-path /path/to/datasets \
  --trust-remote-code --max-model-len 4096 --limit-mm-per-prompt image=1 