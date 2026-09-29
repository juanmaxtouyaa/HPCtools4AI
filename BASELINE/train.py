import argparse
import json
import os
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
        "--torch-compile",
        action="store_true",
        help="Enable torch.compile through Hugging Face TrainingArguments.",
    )

    tf32_group = parser.add_mutually_exclusive_group()

    tf32_group.add_argument(
        "--tf32",
        action="store_true",
        help="Explicitly enable TF32 for FP32 CUDA matrix multiplications.",
    )

    tf32_group.add_argument(
        "--no-tf32",
        action="store_true",
        help="Explicitly disable TF32 for FP32 CUDA matrix multiplications.",
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
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA build: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available.")

    visible_gpu_count = torch.cuda.device_count()

    print(
        f"CUDA_VISIBLE_DEVICES: "
        f"{os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )
    print(f"Visible GPU count: {visible_gpu_count}")

    if visible_gpu_count != 1:
        raise RuntimeError(
            f"Expected exactly 1 visible GPU, "
            f"but found {visible_gpu_count}."
        )

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    if args.tf32:
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        tf32_mode = "on"
    elif args.no_tf32:
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        tf32_mode = "off"
    else:
        tf32_mode = "default"

    print(
        "CUDA matmul FP32 precision: "
        f"{torch.backends.cuda.matmul.fp32_precision}"
    )

    print("\n=== Configuration ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Max sequence length: {MAX_LENGTH}")
    print(f"Document stride: {DOC_STRIDE}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(
        f"DataLoader workers: "
        f"{args.dataloader_num_workers}"
    )
    print(f"BF16: {args.bf16}")
    print(f"TF32 mode: {tf32_mode}")
    print(f"torch.compile: {args.torch_compile}")

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

    print(
        f"Original training examples: "
        f"{len(train_dataset)}"
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

            answer_start_char = answers["answer_start"][0]

            answer_end_char = (
                answer_start_char
                + len(answers["text"][0])
            )

            token_start_index = 0

            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            token_end_index = len(input_ids) - 1

            while sequence_ids[token_end_index] != 1:
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

        tokenized["start_positions"] = start_positions
        tokenized["end_positions"] = end_positions

        return tokenized

    print("\n=== Tokenizing dataset ===")

    tokenized_train = train_dataset.map(
        preprocess_examples,
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    print(
        f"Tokenized training features: "
        f"{len(tokenized_train)}"
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
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_strategy="steps",
        logging_steps=50,
        report_to=["tensorboard"],
        run_name="bert-squad-baseline",
        save_strategy="no",
        eval_strategy="no",
        seed=42,
        dataloader_num_workers=args.dataloader_num_workers,
        bf16=args.bf16,
        torch_compile=args.torch_compile,
        torch_compile_backend=(
            "inductor" if args.torch_compile else None
        ),
        torch_compile_mode=(
            "default" if args.torch_compile else None
        ),
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

    elapsed_seconds = end_time - start_time

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
        f"Measured training time: "
        f"{elapsed_seconds:.2f} seconds"
    )

    print(
        f"Measured training time: "
        f"{elapsed_seconds / 60:.2f} minutes"
    )

    print(
        f"Peak GPU memory allocated: "
        f"{peak_allocated:.2f} GiB"
    )

    print(
        f"Peak GPU memory reserved: "
        f"{peak_reserved:.2f} GiB"
    )

    if not args.skip_save:
        print("\n=== Saving model ===")

        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    results = {
        "model": MODEL_NAME,
        "gpu": torch.cuda.get_device_name(0),
        "visible_gpu_count": visible_gpu_count,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "dataloader_num_workers": (
            args.dataloader_num_workers
        ),
        "bf16": args.bf16,
        "torch_compile": args.torch_compile,
        "torch_compile_backend": (
            "inductor" if args.torch_compile else None
        ),
        "torch_compile_mode": (
            "default" if args.torch_compile else None
        ),
        "tf32_mode": tf32_mode,
        "cuda_matmul_fp32_precision": (
            torch.backends.cuda.matmul.fp32_precision
        ),
        "max_length": MAX_LENGTH,
        "doc_stride": DOC_STRIDE,
        "original_train_examples": (
            len(train_dataset)
        ),
        "tokenized_train_features": (
            len(tokenized_train)
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
        "model_saved": not args.skip_save,
        "trainer_metrics": train_result.metrics,
    }

    results_path = os.path.join(
        args.output_dir,
        "training_results.json",
    )

    with open(results_path, "w") as f:
        json.dump(
            results,
            f,
            indent=4,
        )

    print(
        f"\nResults written to: "
        f"{results_path}"
    )


if __name__ == "__main__":
    main()

