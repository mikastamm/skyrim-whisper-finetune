"""Whisper full fine‑tune script (no LoRA)
=================================================
This script fine‑tunes OpenAI Whisper on custom data while keeping the
 * encoder frozen and
 * the first three decoder blocks frozen.
All remaining decoder weights, token embeddings and output projection are
trainable.

It preserves the argument interface of the original PEFT script but drops
LoRA/AdaLoRA‑specific flags.
"""
import argparse
import functools
import os
import platform

from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from utils.data_utils import DataCollatorSpeechSeq2SeqWithPadding
from utils.model_utils import load_from_checkpoint
from utils.reader import CustomDataset
from utils.utils import print_arguments, add_arguments

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Fine‑tune Whisper on custom data (full fine‑tune, no LoRA)")
add_arg = functools.partial(add_arguments, argparser=parser)
# Dataset & model paths
add_arg("train_data",    type=str, default="dataset/train.json",       help="Path to the training dataset")
add_arg("test_data",     type=str, default="dataset/test.json",        help="Path to the test dataset")
add_arg("base_model",    type=str, default="openai/whisper-base.en",  help="Base Whisper model")
add_arg("output_dir",    type=str, default="output/",                help="Directory to save checkpoints")

# Training hyper‑parameters
add_arg("num_train_epochs",           type=int,   default=8,     help="Number of epochs")
add_arg("per_device_train_batch_size", type=int,   default=24,     help="Train batch size")
add_arg("per_device_eval_batch_size",  type=int,   default=12,     help="Eval batch size")
add_arg("gradient_accumulation_steps", type=int,   default=4,     help="Gradient accumulation steps")
add_arg("learning_rate",               type=float, default=5e-6,  help="Learning rate")
add_arg("warmup_steps",                type=int,   default=1500,  help="Warm‑up steps")
add_arg("logging_steps",               type=int,   default=100,   help="Steps between logging")
add_arg("eval_steps",                  type=int,   default=500,  help="Evaluation frequency (in steps)")
add_arg("save_steps",                  type=int,   default=1000,  help="Checkpoint save frequency")
add_arg("save_total_limit",            type=int,   default=10,    help="Max checkpoints to keep")
add_arg("generation_max_length",       type=int,   default=225,   help="Maximum generation length during eval")
add_arg("weight_decay",                type=float, default=0.0075, help="Weight decay")
add_arg("gradient_checkpointing",      type=bool,  default=True,  help="Enable gradient checkpointing")

# Audio / text processing
add_arg("min_audio_len",   type=float, default=0.5,  help="Minimum audio length (s)")
add_arg("max_audio_len",   type=float, default=30.0, help="Maximum audio length (s)")
add_arg("language",        type=str,   default="English", choices=[None, "English"], help="Language name or code")
add_arg("task",            type=str,   default="transcribe", choices=["transcribe", "translate"], help="Task")
add_arg("timestamps",      type=bool,  default=False, help="Train with timestamp targets")
add_arg("augment_config_path", type=str, default=None, help="Path to data‑augmentation YAML (optional)")

# System / misc
add_arg("num_workers",   type=int,  default=8,    help="Data‑loading workers")
add_arg("fp16",          type=bool, default=True, help="Use fp16 training")
add_arg("use_compile",   type=bool, default=False, help="Use PyTorch 2.0 compile")
add_arg("local_files_only", type=bool, default=False, help="Offline mode – do not download models")

# Checkpoint / hub
add_arg("resume_from_checkpoint", type=str,  default=None, help="Path to resume checkpoint")
add_arg("push_to_hub",           type=bool, default=False, help="Push final model to HF Hub")
add_arg("hub_model_id",          type=str,  default=None, help="HF model repo ID")


# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------

def _fix_windows_workers(args):
    """Set num_workers to 0 on Windows to avoid spawn issues."""
    if platform.system() == "Windows":
        args.num_workers = 0


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    args = parser.parse_args()
    print_arguments(args)
    _fix_windows_workers(args)

    # ------------------------ Processor ------------------------
    processor = WhisperProcessor.from_pretrained(
        args.base_model,
        language=args.language,
        task=args.task,
        no_timestamps=not args.timestamps,
        local_files_only=args.local_files_only,
    )

    # ------------------------ Datasets -------------------------
    train_dataset = CustomDataset(
        data_list_path=args.train_data,
        processor=processor,
        language=args.language,
        timestamps=args.timestamps,
        min_duration=args.min_audio_len,
        max_duration=args.max_audio_len,
        augment_config_path=args.augment_config_path,
    )
    test_dataset = CustomDataset(
        data_list_path=args.test_data,
        processor=processor,
        language=args.language,
        timestamps=args.timestamps,
        min_duration=args.min_audio_len,
        max_duration=args.max_audio_len,
    )
    print(f"Train: {len(train_dataset)}, Test：{len(test_dataset)}")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # -------------------- Model initialisation -----------------
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size != 1:  # DDP
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model,
        load_in_8bit=False,
        device_map=device_map,
        local_files_only=args.local_files_only,
    )

    # Rely on Trainer's built‑in AMP; do **not** convert weights manually
    # (manual .half() causes FP16 gradient unscale errors)
    # Freeze encoder
    print("Freezing all encoder parameters …")
    for p in model.model.encoder.parameters():
        p.requires_grad = False

    # Freeze first N decoder layers
    freeze_decoder_layers = 1
    print(f"Freezing first {freeze_decoder_layers} decoder blocks …")
    for layer in model.model.decoder.layers[:freeze_decoder_layers]:
        for p in layer.parameters():
            p.requires_grad = False

    # Show trainable param counts (optional diagnostics)
    embed_trainable = sum(p.numel() for p in model.model.decoder.embed_tokens.parameters() if p.requires_grad)
    if hasattr(model, "proj_out"):
        lm_head_trainable = sum(p.numel() for p in model.proj_out.parameters() if p.requires_grad)
    else:
        lm_head_trainable = sum(p.numel() for p in model.lm_head.parameters() if p.requires_grad)
    print(f"Token‑embedding trainable params : {embed_trainable:,}")
    print(f"Output‑projection trainable params: {lm_head_trainable:,}")

    # ---------------------- Training args ----------------------
    output_dir = os.path.join(args.output_dir, os.path.basename(args.base_model.rstrip("/")))

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        save_strategy="steps",
        eval_strategy="steps",
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
        fp16=args.fp16,
        report_to=["tensorboard"],
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,
        label_names=["labels"],
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        torch_compile=args.use_compile,
        dataloader_num_workers=args.num_workers,
    )

    # ------------------------- Trainer -------------------------
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
    )

    # custom checkpoint loader from utils
    trainer._load_from_checkpoint = load_from_checkpoint

    # Disable cache during training for speed / memory
    model.config.use_cache = False

    # ------------------------- Train ---------------------------
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state()

    # Re‑enable cache for inference and save final model
    model.config.use_cache = True
    if training_args.local_rank in (-1, 0):
        model.save_pretrained(os.path.join(output_dir, "checkpoint-final"))

    # Optional: push to Hub
    if training_args.push_to_hub and training_args.local_rank in (-1, 0):
        hub_repo = args.hub_model_id or output_dir
        model.push_to_hub(hub_repo)

    print("\n✅ Training complete!")
