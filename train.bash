#!/usr/bin/env bash
set -euo pipefail

# ----- tweak these two lines when you want to switch model / language -----
MODEL="whisper-large-v3-turbo"     # base HF model name minus the “openai/” prefix
LANGUAGE="English"       # language name or BCP-47 code
# -------------------------------------------------------------------------

CUDA_VISIBLE_DEVICES=0 python finetune.py \
  --base_model "openai/${MODEL}" \
    --language    "${LANGUAGE}" \

# We need no more merging as we do a full finetune
# python merge_lora.py \
#  --lora_model "output/${MODEL}/checkpoint-best/" 

python eval-batch.py \
  --model    "${MODEL}" \
  --language "${LANGUAGE}"

python convert-ggml.py \
    --model_dir "output/${MODEL}/checkpoint-final" \
    --output_path "models/ggml-skyrim-${MODEL}.bin" 