import argparse
import functools
import os
import platform

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, WhisperProcessor
from utils.callback import SavePeftModelCallback
from utils.data_utils import DataCollatorSpeechSeq2SeqWithPadding
from utils.model_utils import load_from_checkpoint
from utils.reader import CustomDataset
from utils.utils import print_arguments, make_inputs_require_grad, add_arguments

# -------------------- Argument Definitions --------------------
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("train_data",    type=str, default="dataset/train.json",       help="Path to the training dataset")
add_arg("test_data",     type=str, default="dataset/test.json",        help="Path to the test dataset")
add_arg("base_model",    type=str, default="openai/whisper-base.en",  help="Base Whisper model")
add_arg("output_dir",    type=str, default="models/",                help="Path to save the trained model")
add_arg("warmup_steps",  type=int, default=50,                        help="Number of warmup steps")
add_arg("logging_steps", type=int, default=100,                       help="Steps between logging")
add_arg("eval_steps",    type=int, default=1000,                      help="Evaluation frequency (in steps)")
add_arg("save_steps",    type=int, default=1000,                      help="Model save frequency (in steps)")
add_arg("num_workers",   type=int, default=8,                        help="Number of data loading threads")
add_arg("learning_rate", type=float, default=5e-6,                    help="Learning rate")
add_arg("min_audio_len", type=float, default=0.5,                     help="Minimum audio length in seconds")
add_arg("max_audio_len", type=float, default=30,                      help="Maximum audio length in seconds (max 30s)")
add_arg("fp16",          type=bool,  default=True,                  help="Use fp16 for model training")
add_arg("timestamps",    type=bool,  default=False,                 help="Use timestamp data during training")
add_arg("use_compile",   type=bool,  default=False,                 help="Use PyTorch 2.0 compiler")
add_arg("local_files_only", type=bool, default=False,               help="Load model only from local files, no downloads")
add_arg("num_train_epochs", type=int, default=4,                    help="Number of training epochs")
add_arg("language",      type=str, default="English",              help="Set language (full name or abbreviation), or None for multilingual")
add_arg("task",          type=str, default="transcribe",           choices=['transcribe', 'translate'], help="Model task")
add_arg("augment_config_path",    type=str, default=None,           help="Path to data augmentation config file")
add_arg("resume_from_checkpoint", type=str, default=None,           help="Path to resume training from a checkpoint")
add_arg("per_device_train_batch_size", type=int, default=8,         help="Batch size for training")
add_arg("per_device_eval_batch_size",  type=int, default=8,         help="Batch size for evaluation")
add_arg("gradient_accumulation_steps", type=int, default=1,         help="Gradient accumulation steps")
add_arg("push_to_hub",               type=bool, default=False,     help="Push model weights to HuggingFace Hub")
add_arg("hub_model_id",              type=str,  default=None,      help="Model repository ID on HuggingFace Hub")
add_arg("save_total_limit",          type=int,  default=10,        help="Maximum number of checkpoints to keep")

# Windows fallback for num_workers
def _fix_windows_workers(args):
    if platform.system() == "Windows":
        args.num_workers = 0

if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)
    _fix_windows_workers(args)

    # ----------------------------------------------------------------------
    # Load processor
    # ----------------------------------------------------------------------
    processor = WhisperProcessor.from_pretrained(
        args.base_model,
        language=args.language,
        task=args.task,
        no_timestamps=not args.timestamps,
        local_files_only=args.local_files_only
    )

    # Load datasets
    train_dataset = CustomDataset(
        data_list_path=args.train_data,
        processor=processor,
        language=args.language,
        timestamps=args.timestamps,
        min_duration=args.min_audio_len,
        max_duration=args.max_audio_len,
        augment_config_path=args.augment_config_path
    )
    test_dataset = CustomDataset(
        data_list_path=args.test_data,
        processor=processor,
        language=args.language,
        timestamps=args.timestamps,
        min_duration=args.min_audio_len,
        max_duration=args.max_audio_len
    )
    print(f"训练数据：{len(train_dataset)}，测试数据：{len(test_dataset)}")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # ----------------------------------------------------------------------
    # MODEL INITIALIZATION: Full fine-tuning regime (no LoRA)
    # ----------------------------------------------------------------------
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model,
        load_in_8bit=False,
        device_map=device_map,
        local_files_only=args.local_files_only
    )

    if args.fp16:
        model = model.half()

    # Freeze encoder
    print("Freezing all encoder parameters …")
    for p in model.model.encoder.parameters():
        p.requires_grad = False

    # Freeze first N decoder layers
    freeze_decoder_layers = 3
    print(f"Freezing first {freeze_decoder_layers} decoder blocks …")
    for layer in model.model.decoder.layers[:freeze_decoder_layers]:
        for p in layer.parameters():
            p.requires_grad = False

    # Confirm embeddings and lm_head are trainable
    enum = sum(p.numel() for p in model.model.decoder.embed_tokens.parameters() if p.requires_grad)
    lnum = sum(p.numel() for p in model.lm_head.parameters() if p.requires_grad)
    print(f"Token-embedding params trainable: {enum:,}")
    print(f"LM-head params trainable       : {lnum:,}")

    callbacks = []

    # ----------------------------------------------------------------------
    # TRAINING ARGUMENTS
    # ----------------------------------------------------------------------
    output_dir = os.path.join(args.output_dir, os.path.basename(args.base_model.rstrip('/')))
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
        load_best_model_at_end=True,
        fp16=args.fp16,
        report_to=["tensorboard"],
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,
        label_names=["labels"],
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )

    # ----------------------------------------------------------------------
    # SEQ2SEQ TRAINER
    # ----------------------------------------------------------------------
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
        processing_class=processor.feature_extractor
    )
    model.config.use_cache = False
    trainer._load_from_checkpoint = load_from_checkpoint

    # ----------------------------------------------------------------------
    # TRAIN & SAVE
    # ----------------------------------------------------------------------
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state()
    model.config.use_cache = True
    if training_args.local_rank in (-1, 0):
        model.save_pretrained(os.path.join(output_dir, "checkpoint-final"))
    if training_args.push_to_hub:
        hub_id = args.hub_model_id or output_dir
        model.push_to_hub(hub_id)

    print("Training complete!")
