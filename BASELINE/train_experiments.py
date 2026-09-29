import argparse
import json
import math
import os
import socket
import time

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)


MODEL_NAME = "google-bert/bert-base-uncased"
MAX_LENGTH = 384
DOC_STRIDE = 128


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-5,
    )

    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--bf16",
        action="store_true",
    )

    parser.add_argument(
        "--drop-last",
        action="store_true",
        help="Drop the final incomplete training batch.",
    )

    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Enable torch.compile through TrainingArguments.",
    )

    parser.add_argument(
        "--torch-compile-mode",
        type=str,
        default="default",
        choices=[
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ],
        help="torch.compile mode when compilation is enabled.",
    )

    tf32_group = parser.add_mutually_exclusive_group()

    tf32_group.add_argument(
        "--tf32",
        action="store_true",
        help="Enable TF32 for CUDA FP32 matrix multiplications.",
    )

    tf32_group.add_argument(
        "--no-tf32",
        action="store_true",
        help="Disable TF32 for CUDA FP32 matrix multiplications.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--preprocess-only",
        action="store_true",
    )

    parser.add_argument(
        "--skip-save",
        action="store_true",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/baseline",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=== Environment ===")
    print(f"Host: {socket.gethostname()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA build: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available.")

    visible_gpu_count = torch.cuda.device_count()

    print(
        "CUDA_VISIBLE_DEVICES: "
        f"{os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )
    print(f"Visible GPU count: {visible_gpu_count}")

    if visible_gpu_count != 1:
        raise RuntimeError(
            "Expected exactly 1 visible GPU, "
            f"but found {visible_gpu_count}."
        )

    gpu_name = torch.cuda.get_device_name(0)

    print(f"GPU: {gpu_name}")

    # Explicit TF32 control.
    #
    # No flag:
    #     preserve the current PyTorch/default behavior.
    #
    # --tf32:
    #     request TF32 for FP32 CUDA matmul.
    #
    # --no-tf32:
    #     request IEEE FP32 matmul.
    if args.tf32:
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        tf32_mode = "on"
    elif args.no_tf32:
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        tf32_mode = "off"
    else:
        tf32_mode = "default"

    cuda_matmul_fp32_precision = str(
        torch.backends.cuda.matmul.fp32_precision
    )

    print(
        "CUDA matmul FP32 precision: "
        f"{cuda_matmul_fp32_precision}"
    )

    print("\n=== Configuration ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Max sequence length: {MAX_LENGTH}")
    print(f"Document stride: {DOC_STRIDE}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(
        "DataLoader workers: "
        f"{args.dataloader_num_workers}"
    )
    print(f"BF16: {args.bf16}")
    print(f"Drop last: {args.drop_last}")
    print(f"TF32 mode: {tf32_mode}")
    print(f"torch.compile: {args.torch_compile}")

    if args.torch_compile:
        print(
            "torch.compile backend: inductor"
        )
        print(
            "torch.compile mode: "
            f"{args.torch_compile_mode}"
        )

    print(f"Seed: {args.seed}")

    print("\n=== Loading tokenizer and model ===")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    model = AutoModelForQuestionAnswering.from_pretrained(
        MODEL_NAME
    )

    print("\n=== Loading SQuAD ===")

    dataset = load_dataset("rajpurkar/squad")
    train_dataset = dataset["train"]

    if args.max_train_samples is not None:
        number_of_samples = min(
            args.max_train_samples,
            len(train_dataset),
        )

        train_dataset = train_dataset.select(
            range(number_of_samples)
        )

    original_train_examples = len(train_dataset)

    print(
        "Original training examples: "
        f"{original_train_examples}"
    )

    def preprocess_examples(examples):
        questions = [
            question.strip()
            for question in examples["question"]
        ]

        tokenized = tokenizer(
            questions,
            examples["context"],
            max_length=MAX_LENGTH,
            truncation="only_second",
            stride=DOC_STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_mapping = tokenized.pop(
            "overflow_to_sample_mapping"
        )

        offset_mapping = tokenized.pop(
            "offset_mapping"
        )

        start_positions = []
        end_positions = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized["input_ids"][i]

            cls_index = input_ids.index(
                tokenizer.cls_token_id
            )

            sequence_ids = tokenized.sequence_ids(i)

            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]

            if len(answers["answer_start"]) == 0:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
                continue

            answer_start_char = answers[
                "answer_start"
            ][0]

            answer_end_char = (
                answer_start_char
                + len(answers["text"][0])
            )

            token_start_index = 0

            while (
                sequence_ids[token_start_index] != 1
            ):
                token_start_index += 1

            token_end_index = len(input_ids) - 1

            while (
                sequence_ids[token_end_index] != 1
            ):
                token_end_index -= 1

            if (
                offsets[token_start_index][0]
                > answer_start_char
                or offsets[token_end_index][1]
                < answer_end_char
            ):
                start_positions.append(cls_index)
                end_positions.append(cls_index)
                continue

            while (
                token_start_index < len(offsets)
                and offsets[token_start_index][0]
                <= answer_start_char
            ):
                token_start_index += 1

            start_positions.append(
                token_start_index - 1
            )

            while (
                token_end_index >= 0
                and offsets[token_end_index][1]
                >= answer_end_char
            ):
                token_end_index -= 1

            end_positions.append(
                token_end_index + 1
            )

        tokenized[
            "start_positions"
        ] = start_positions

        tokenized[
            "end_positions"
        ] = end_positions

        return tokenized

    print("\n=== Tokenizing dataset ===")

    tokenized_train = train_dataset.map(
        preprocess_examples,
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    tokenized_train_features = len(
        tokenized_train
    )

    print(
        "Tokenized training features: "
        f"{tokenized_train_features}"
    )

    full_batches = (
        tokenized_train_features
        // args.batch_size
    )

    tail_batch_size = (
        tokenized_train_features
        % args.batch_size
    )

    if args.drop_last:
        expected_batches_per_epoch = full_batches
        effective_features_per_epoch = (
            full_batches * args.batch_size
        )
    else:
        expected_batches_per_epoch = math.ceil(
            tokenized_train_features
            / args.batch_size
        )
        effective_features_per_epoch = (
            tokenized_train_features
        )

    dropped_features_per_epoch = (
        tokenized_train_features
        - effective_features_per_epoch
    )

    print("\n=== Batch geometry ===")
    print(f"Full batches: {full_batches}")
    print(f"Tail batch size: {tail_batch_size}")
    print(
        "Expected batches/epoch: "
        f"{expected_batches_per_epoch}"
    )
    print(
        "Effective features/epoch: "
        f"{effective_features_per_epoch}"
    )
    print(
        "Dropped features/epoch: "
        f"{dropped_features_per_epoch}"
    )

    if args.preprocess_only:
        print(
            "\n=== Preprocessing finished; "
            "training skipped ==="
        )
        return

    data_collator = DefaultDataCollator()

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,

        num_train_epochs=args.epochs,

        per_device_train_batch_size=(
            args.batch_size
        ),

        learning_rate=args.learning_rate,
        weight_decay=0.01,

        # Make previously implicit choices explicit
        # for experiment reproducibility.
        optim="adamw_torch_fused",
        lr_scheduler_type="linear",
        gradient_accumulation_steps=1,

        logging_strategy="steps",
        logging_steps=50,
        report_to=["tensorboard"],
        run_name="bert-squad-experiment",

        save_strategy="no",
        eval_strategy="no",

        seed=args.seed,
        data_seed=args.seed,

        dataloader_num_workers=(
            args.dataloader_num_workers
        ),
        dataloader_pin_memory=True,
        dataloader_drop_last=args.drop_last,

        bf16=args.bf16,

        torch_compile=args.torch_compile,

        torch_compile_backend=(
            "inductor"
            if args.torch_compile
            else None
        ),

        torch_compile_mode=(
            args.torch_compile_mode
            if args.torch_compile
            else None
        ),
    )

    print("\n=== Resolved Trainer configuration ===")
    print(f"Optimizer: {training_args.optim}")
    print(
        "LR scheduler: "
        f"{training_args.lr_scheduler_type}"
    )
    print(
        "Gradient accumulation steps: "
        f"{training_args.gradient_accumulation_steps}"
    )
    print(
        "DataLoader pin memory: "
        f"{training_args.dataloader_pin_memory}"
    )
    print(
        "DataLoader drop last: "
        f"{training_args.dataloader_drop_last}"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print("\n=== Starting training ===")

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()

    train_result = trainer.train()

    torch.cuda.synchronize()

    end_time = time.perf_counter()

    elapsed_seconds = (
        end_time - start_time
    )

    peak_allocated = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    peak_reserved = (
        torch.cuda.max_memory_reserved()
        / 1024**3
    )

    print("\n=== Training finished ===")

    print(
        "Measured training time: "
        f"{elapsed_seconds:.2f} seconds"
    )

    print(
        "Measured training time: "
        f"{elapsed_seconds / 60:.2f} minutes"
    )

    print(
        "Peak GPU memory allocated: "
        f"{peak_allocated:.2f} GiB"
    )

    print(
        "Peak GPU memory reserved: "
        f"{peak_reserved:.2f} GiB"
    )

    if not args.skip_save:
        print("\n=== Saving model ===")

        trainer.save_model(
            args.output_dir
        )

        tokenizer.save_pretrained(
            args.output_dir
        )

    expected_total_steps = None

    if float(args.epochs).is_integer():
        expected_total_steps = (
            expected_batches_per_epoch
            * int(args.epochs)
        )

    results = {
        "model": MODEL_NAME,

        "host": socket.gethostname(),

        "slurm_job_id": os.environ.get(
            "SLURM_JOB_ID"
        ),

        "slurm_array_task_id": os.environ.get(
            "SLURM_ARRAY_TASK_ID"
        ),

        "slurm_cpus_per_task": os.environ.get(
            "SLURM_CPUS_PER_TASK"
        ),

        "cuda_visible_devices": os.environ.get(
            "CUDA_VISIBLE_DEVICES"
        ),

        "gpu": gpu_name,

        "visible_gpu_count": (
            visible_gpu_count
        ),

        "pytorch_version": (
            torch.__version__
        ),

        "cuda_version": (
            torch.version.cuda
        ),

        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,

        "dataloader_num_workers": (
            args.dataloader_num_workers
        ),

        "dataloader_pin_memory": (
            training_args.dataloader_pin_memory
        ),

        "drop_last": args.drop_last,

        "gradient_accumulation_steps": (
            training_args.gradient_accumulation_steps
        ),

        "optimizer": str(
            training_args.optim
        ),

        "lr_scheduler_type": str(
            training_args.lr_scheduler_type
        ),

        "bf16": args.bf16,

        "tf32_mode": tf32_mode,

        "cuda_matmul_fp32_precision": (
            cuda_matmul_fp32_precision
        ),

        "torch_compile": (
            args.torch_compile
        ),

        "torch_compile_backend": (
            "inductor"
            if args.torch_compile
            else None
        ),

        "torch_compile_mode": (
            args.torch_compile_mode
            if args.torch_compile
            else None
        ),

        "seed": args.seed,

        "max_length": MAX_LENGTH,
        "doc_stride": DOC_STRIDE,

        "original_train_examples": (
            original_train_examples
        ),

        "tokenized_train_features": (
            tokenized_train_features
        ),

        "full_batches_per_epoch": (
            full_batches
        ),

        "tail_batch_size": (
            tail_batch_size
        ),

        "expected_batches_per_epoch": (
            expected_batches_per_epoch
        ),

        "effective_features_per_epoch": (
            effective_features_per_epoch
        ),

        "dropped_features_per_epoch": (
            dropped_features_per_epoch
        ),

        "expected_total_steps": (
            expected_total_steps
        ),

        "training_time_seconds": (
            elapsed_seconds
        ),

        "training_time_minutes": (
            elapsed_seconds / 60
        ),

        "peak_gpu_memory_allocated_gib": (
            peak_allocated
        ),

        "peak_gpu_memory_reserved_gib": (
            peak_reserved
        ),

        "model_saved": (
            not args.skip_save
        ),

        "trainer_metrics": (
            train_result.metrics
        ),
    }

    results_path = os.path.join(
        args.output_dir,
        "training_results.json",
    )

    with open(
        results_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=4,
        )

    print(
        "\nResults written to: "
        f"{results_path}"
    )


if __name__ == "__main__":
    main()

