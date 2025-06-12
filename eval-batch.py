#!/usr/bin/env python3
"""
evaluation.py  –  minimal, multi-run evaluation launcher
--------------------------------------------------------

Usage
-----
python evaluation.py --model whisper-tiny --language English
"""

import argparse
import functools
import gc
import os
import platform
from pathlib import Path
from typing import List, Tuple

import evaluate
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from utils.data_utils import (
    DataCollatorSpeechSeq2SeqWithPadding,
    remove_punctuation,
    to_simple,
)
from utils.reader import CustomDataset
from utils.utils import add_arguments, print_arguments


# --------------------------------------------------------------------------- #
# -------------------------  static “hyper-parameters”  ---------------------- #
# --------------------------------------------------------------------------- #
BATCH_SIZE: int = 24
NUM_WORKERS: int = 0 if platform.system() == "Windows" else 8
MIN_AUDIO_S: float = 0.5
MAX_AUDIO_S: float = 30.0
REMOVE_PUN: bool = True
TO_SIMPLE_CN: bool = True
TIMESTAMPS: bool = False
TASK: str = "transcribe"


# --------------------------------------------------------------------------- #
# -------------------------  Core evaluation logic  ------------------------- #
# --------------------------------------------------------------------------- #
def evaluate_once(
    *,
    test_data: str | Path,
    model_path: str | Path,
    processor_src: str | Path,              # <-- NEW: where to get tokenizer/FE
    metric_name: str,
    language: str,
    local_files_only: bool,
) -> float:
    """Run a single evaluation and return the metric value."""
    # —- processor always loaded from base model —
    processor = WhisperProcessor.from_pretrained(
        processor_src,
        language=language,
        task=TASK,
        no_timestamps=not TIMESTAMPS,
        local_files_only=local_files_only,
    )

    # —- model weights from the specific checkpoint/path —
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        local_files_only=True,               # always local for both cases below
    )
    model.generation_config.forced_decoder_ids = None
    model.eval()

    dataset = CustomDataset(
        data_list_path=str(test_data),
        processor=processor,
        timestamps=TIMESTAMPS,
        min_duration=MIN_AUDIO_S,
        max_duration=MAX_AUDIO_S,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        collate_fn=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
    )

    metric = evaluate.load(f"metrics/{metric_name}.py")

    for batch in tqdm(
        dataloader,
        desc=f"[{metric_name}] {Path(str(model_path)).name}",
        leave=False,
    ):
        with torch.autocast(device_type="cuda"):
            with torch.no_grad():
                generated = (
                    model.generate(
                        input_features=batch["input_features"].cuda(),
                        decoder_input_ids=batch["labels"][:, :4].cuda(),
                        max_new_tokens=255,
                    )
                    .cpu()
                    .numpy()
                )

        labels = batch["labels"].cpu().numpy()
        labels = np.where(
            labels != -100, labels, processor.tokenizer.pad_token_id
        )

        preds_text = processor.tokenizer.batch_decode(
            generated, skip_special_tokens=True
        )
        labels_text = processor.tokenizer.batch_decode(
            labels, skip_special_tokens=True
        )

        if REMOVE_PUN:
            preds_text = remove_punctuation(preds_text)
            labels_text = remove_punctuation(labels_text)
        if TO_SIMPLE_CN:
            preds_text = to_simple(preds_text)
            labels_text = to_simple(labels_text)

        metric.add_batch(predictions=preds_text, references=labels_text)

        # cleanup
        del generated, labels, batch
        gc.collect()

    return round(metric.compute(), 5)


# --------------------------------------------------------------------------- #
# ------------------------------  CLI driver  --------------------------------#
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser("Whisper evaluation utility")
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg(
        "model",
        type=str,
        default="whisper-xxx",
        help="base model name (e.g. whisper-tiny, whisper-small)",
    )
    add_arg(
        "language",
        type=str,
        default="English",
        help="language name or code (e.g. English or en)",
    )
    args = parser.parse_args()
    print_arguments(args)

    # ------------------------------------------------------- #
    # test-set / metric combinations to run in every session  #
    # ------------------------------------------------------- #
    combos: List[Tuple[str, str]] = [
        ("dataset/test-skyrim.json", "skyrim"),
      #  ("dataset/test-skyrim.json", "wer"),
        ("dataset/test-commonvoice.json", "wer"),
    ]

    # ------------------------------------------------------- #
    # base vs. fine-tuned                                     #
    # ------------------------------------------------------- #
    results = []
    for label in ("BASE", "FINETUNED"):
        if label == "FINETUNED":
            # weights path
            model_path = Path(f"output/{args.model}/checkpoint-final")
            # processor from original base model
            processor_src = f"openai/{args.model}"
            local_only = True
        else:  # BASE
            local_cp = Path(f"models/{args.model}")
            if local_cp.exists():
                model_path = processor_src = local_cp
                local_only = True
            else:
                model_path = processor_src = f"openai/{args.model}"
                local_only = False  
        for test_file, metric in combos:
            score = evaluate_once(
                test_data=test_file,
                model_path=model_path,
                processor_src=processor_src,
                metric_name=metric,
                language=args.language,
                local_files_only=local_only,
            )
            results.append(
                {
                    "Model": label,
                    "Metric": metric,
                    "Test-set": Path(test_file).stem,
                    "Score": score,
                }
            )


    # ------------------------------------------------------- #
    #                Nicely format the results                #
    # ------------------------------------------------------- #
    print("\n\n==========  FINAL RESULTS  ==========")
    header = "{:<10} │ {:<7} │ {:<17} │ {:>8}"
    row_fmt = "{:<10} │ {:<7} │ {:<17} │ {:>8.5f}"
    print(header.format("Model", "Metric", "Test-set", "Score"))
    print("───────────┼─────────┼───────────────────┼──────────")
    for r in results:
        print(
            row_fmt.format(
                r["Model"], r["Metric"], r["Test-set"], r["Score"]
            )
        )
    print("=====================================\n")


if __name__ == "__main__":
    main()
