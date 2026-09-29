import argparse
import json
import math
import os
import re
import socket
import string
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DefaultDataCollator,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
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
        default=10.0,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=160,
    )

    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=160,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-5,
    )

    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=32768,
    )

    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--bf16",
        action="store_true",
    )

    parser.add_argument(
        "--drop-last",
        action="store_true",
    )

    parser.add_argument(
        "--torch-compile",
        action="store_true",
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
    )

    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--early-stopping-threshold",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    return parser.parse_args()


def write_phase(path, phase):
    Path(path).write_text(
        phase + "\n",
        encoding="utf-8",
    )


def preprocess_examples(
    examples,
    tokenizer,
):
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

        answer_start_char = (
            answers["answer_start"][0]
        )

        answer_end_char = (
            answer_start_char
            + len(answers["text"][0])
        )

        token_start_index = 0

        while (
            sequence_ids[token_start_index] != 1
        ):
            token_start_index += 1

        token_end_index = (
            len(input_ids) - 1
        )

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


def prepare_validation_features(
    examples,
    tokenizer,
):
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

    example_ids = []

    for i in range(
        len(tokenized["input_ids"])
    ):
        sequence_ids = tokenized.sequence_ids(i)

        sample_index = sample_mapping[i]

        example_ids.append(
            examples["id"][sample_index]
        )

        offsets = tokenized[
            "offset_mapping"
        ][i]

        tokenized[
            "offset_mapping"
        ][i] = [
            offset
            if sequence_ids[k] == 1
            else None
            for k, offset in enumerate(offsets)
        ]

    tokenized["example_id"] = example_ids

    return tokenized


def normalize_answer(text):
    def remove_articles(value):
        return re.sub(
            r"\b(a|an|the)\b",
            " ",
            value,
        )

    def remove_punctuation(value):
        return "".join(
            char
            for char in value
            if char not in string.punctuation
        )

    def fix_whitespace(value):
        return " ".join(value.split())

    return fix_whitespace(
        remove_articles(
            remove_punctuation(
                text.lower()
            )
        )
    )


def exact_match_score(
    prediction,
    ground_truth,
):
    return int(
        normalize_answer(prediction)
        == normalize_answer(ground_truth)
    )


def f1_score(
    prediction,
    ground_truth,
):
    prediction_tokens = (
        normalize_answer(prediction).split()
    )

    ground_truth_tokens = (
        normalize_answer(ground_truth).split()
    )

    common = (
        Counter(prediction_tokens)
        & Counter(ground_truth_tokens)
    )

    num_same = sum(common.values())

    if (
        len(prediction_tokens) == 0
        or len(ground_truth_tokens) == 0
    ):
        return float(
            prediction_tokens
            == ground_truth_tokens
        )

    if num_same == 0:
        return 0.0

    precision = (
        num_same
        / len(prediction_tokens)
    )

    recall = (
        num_same
        / len(ground_truth_tokens)
    )

    return (
        2 * precision * recall
        / (precision + recall)
    )


def postprocess_qa_predictions(
    examples,
    features,
    raw_predictions,
    n_best_size=20,
    max_answer_length=30,
):
    start_logits, end_logits = (
        raw_predictions[:2]
    )

    example_id_to_index = {
        example_id: i
        for i, example_id
        in enumerate(examples["id"])
    }

    features_per_example = defaultdict(list)

    for i, feature in enumerate(features):
        example_index = (
            example_id_to_index[
                feature["example_id"]
            ]
        )

        features_per_example[
            example_index
        ].append(i)

    predictions = {}

    for example_index, example in enumerate(
        examples
    ):
        context = example["context"]
        candidates = []

        for feature_index in (
            features_per_example[
                example_index
            ]
        ):
            offsets = features[
                feature_index
            ]["offset_mapping"]

            feature_start_logits = (
                start_logits[feature_index]
            )

            feature_end_logits = (
                end_logits[feature_index]
            )

            start_indexes = np.argsort(
                feature_start_logits
            )[-n_best_size:][::-1]

            end_indexes = np.argsort(
                feature_end_logits
            )[-n_best_size:][::-1]

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index
                        >= len(offsets)
                        or end_index
                        >= len(offsets)
                    ):
                        continue

                    if (
                        offsets[start_index]
                        is None
                        or offsets[end_index]
                        is None
                    ):
                        continue

                    if (
                        end_index
                        < start_index
                    ):
                        continue

                    if (
                        end_index
                        - start_index
                        + 1
                        > max_answer_length
                    ):
                        continue

                    start_char = (
                        offsets[start_index][0]
                    )

                    end_char = (
                        offsets[end_index][1]
                    )

                    score = (
                        feature_start_logits[
                            start_index
                        ]
                        + feature_end_logits[
                            end_index
                        ]
                    )

                    candidates.append(
                        {
                            "score": float(score),
                            "text": context[
                                start_char:end_char
                            ],
                        }
                    )

        if candidates:
            best = max(
                candidates,
                key=lambda x: x["score"],
            )

            predictions[
                example["id"]
            ] = best["text"]
        else:
            predictions[
                example["id"]
            ] = ""

    return predictions


def compute_squad_metrics(
    examples,
    predictions,
):
    exact_match = 0.0
    f1 = 0.0

    for example in examples:
        prediction = predictions[
            example["id"]
        ]

        ground_truths = (
            example["answers"]["text"]
        )

        if not ground_truths:
            ground_truths = [""]

        exact_match += max(
            exact_match_score(
                prediction,
                truth,
            )
            for truth in ground_truths
        )

        f1 += max(
            f1_score(
                prediction,
                truth,
            )
            for truth in ground_truths
        )

    count = len(examples)

    return {
        "exact_match": (
            100.0 * exact_match / count
        ),
        "f1": (
            100.0 * f1 / count
        ),
    }


class ResourceTimingCallback(
    TrainerCallback
):
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)

        self.events_path = (
            self.output_dir
            / "epoch_resource_events.jsonl"
        )

        self.step_start = None
        self.epoch_start = None

        self.training_step_wall_seconds = 0.0

        self.max_allocated_gib_seen = 0.0
        self.max_reserved_gib_seen = 0.0

    def _gpu_stats(self):
        allocated = (
            torch.cuda.memory_allocated()
            / 1024**3
        )

        reserved = (
            torch.cuda.memory_reserved()
            / 1024**3
        )

        peak_allocated = (
            torch.cuda.max_memory_allocated()
            / 1024**3
        )

        peak_reserved = (
            torch.cuda.max_memory_reserved()
            / 1024**3
        )

        self.max_allocated_gib_seen = max(
            self.max_allocated_gib_seen,
            peak_allocated,
        )

        self.max_reserved_gib_seen = max(
            self.max_reserved_gib_seen,
            peak_reserved,
        )

        return {
            "gpu_allocated_gib": allocated,
            "gpu_reserved_gib": reserved,
            "gpu_peak_allocated_gib": (
                peak_allocated
            ),
            "gpu_peak_reserved_gib": (
                peak_reserved
            ),
        }

    def _write(self, record):
        with self.events_path.open(
            "a",
            encoding="utf-8",
        ) as f:
            json.dump(
                record,
                f,
                default=str,
            )

            f.write("\n")

    def on_epoch_begin(
        self,
        args,
        state,
        control,
        **kwargs,
    ):
        torch.cuda.reset_peak_memory_stats()

        self.epoch_start = (
            time.perf_counter()
        )

        return control

    def on_step_begin(
        self,
        args,
        state,
        control,
        **kwargs,
    ):
        self.step_start = (
            time.perf_counter()
        )

        return control

    def on_step_end(
        self,
        args,
        state,
        control,
        **kwargs,
    ):
        if self.step_start is not None:
            self.training_step_wall_seconds += (
                time.perf_counter()
                - self.step_start
            )

        self.step_start = None

        return control

    def on_epoch_end(
        self,
        args,
        state,
        control,
        **kwargs,
    ):
        record = {
            "event": "train_epoch_end",
            "epoch": state.epoch,
            "global_step": state.global_step,
        }

        if self.epoch_start is not None:
            record[
                "epoch_training_wall_seconds"
            ] = (
                time.perf_counter()
                - self.epoch_start
            )

        record.update(
            self._gpu_stats()
        )

        self._write(record)

        return control

    def on_evaluate(
        self,
        args,
        state,
        control,
        metrics=None,
        **kwargs,
    ):
        record = {
            "event": "evaluation",
            "epoch": state.epoch,
            "global_step": state.global_step,
        }

        if metrics:
            record.update(metrics)

        record.update(
            self._gpu_stats()
        )

        self._write(record)

        return control


class InstrumentedTrainer(Trainer):
    def __init__(
        self,
        *args,
        phase_file=None,
        **kwargs,
    ):
        self.phase_file = phase_file

        self.eval_wall_seconds = 0.0
        self.eval_records = []

        self.predict_wall_seconds = 0.0

        self.eval_phase = "validation"
        self.after_eval_phase = "training"

        super().__init__(*args, **kwargs)

    def _phase(self, value):
        if self.phase_file is not None:
            write_phase(
                self.phase_file,
                value,
            )

    def set_eval_phase(
        self,
        phase,
        after,
    ):
        self.eval_phase = phase
        self.after_eval_phase = after

    def get_eval_dataloader(
        self,
        eval_dataset=None,
    ):
        # Important:
        # drop_last belongs to the TRAINING
        # experiment only.
        #
        # Validation always uses every feature.
        original = (
            self.args.dataloader_drop_last
        )

        self.args.dataloader_drop_last = False

        try:
            return super().get_eval_dataloader(
                eval_dataset
            )
        finally:
            self.args.dataloader_drop_last = (
                original
            )

    def get_test_dataloader(
        self,
        test_dataset,
    ):
        original = (
            self.args.dataloader_drop_last
        )

        self.args.dataloader_drop_last = False

        try:
            return super().get_test_dataloader(
                test_dataset
            )
        finally:
            self.args.dataloader_drop_last = (
                original
            )

    def evaluate(
        self,
        *args,
        **kwargs,
    ):
        self._phase(self.eval_phase)

        start = time.perf_counter()

        try:
            metrics = super().evaluate(
                *args,
                **kwargs,
            )
        finally:
            elapsed = (
                time.perf_counter()
                - start
            )

            self.eval_wall_seconds += elapsed

            self._phase(
                self.after_eval_phase
            )

        self.eval_records.append(
            {
                "phase": self.eval_phase,
                "wall_seconds": elapsed,
                "metrics": metrics,
            }
        )

        return metrics

    def predict(
        self,
        *args,
        **kwargs,
    ):
        self._phase("squad_prediction")

        start = time.perf_counter()

        try:
            output = super().predict(
                *args,
                **kwargs,
            )
        finally:
            elapsed = (
                time.perf_counter()
                - start
            )

            self.predict_wall_seconds += (
                elapsed
            )

            self._phase("post_training")

        return output


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_dir = Path(
        os.environ.get(
            "CHECKPOINT_DIR",
            str(output_dir / "checkpoints"),
        )
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Checkpoint directory: {checkpoint_dir}"
    )

    phase_file = (
        output_dir / "phase.txt"
    )

    write_phase(
        phase_file,
        "setup",
    )

    print("=== Environment ===")
    print(f"Host: {socket.gethostname()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA build: {torch.version.cuda}")
    print(
        "CUDA available: "
        f"{torch.cuda.is_available()}"
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not available."
        )

    visible_gpu_count = (
        torch.cuda.device_count()
    )

    print(
        "CUDA_VISIBLE_DEVICES: "
        f"{os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )

    print(
        "Visible GPU count: "
        f"{visible_gpu_count}"
    )

    if visible_gpu_count != 1:
        raise RuntimeError(
            "Expected exactly one visible GPU, "
            f"found {visible_gpu_count}."
        )

    gpu_name = (
        torch.cuda.get_device_name(0)
    )

    print(f"GPU: {gpu_name}")

    print("\n=== Configuration ===")
    print(f"Epoch maximum: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(
        "Eval batch size: "
        f"{args.eval_batch_size}"
    )
    print(
        f"Learning rate: "
        f"{args.learning_rate}"
    )
    print(f"BF16: {args.bf16}")
    print(
        f"Training drop_last: "
        f"{args.drop_last}"
    )
    print(
        "Validation drop_last: False"
    )
    print(
        f"torch.compile: "
        f"{args.torch_compile}"
    )
    print(
        "Early stopping patience: "
        f"{args.early_stopping_patience}"
    )
    print(
        "Early stopping threshold: "
        f"{args.early_stopping_threshold}"
    )

    write_phase(
        phase_file,
        "loading_model",
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME,
            use_fast=True,
        )
    )

    model = (
        AutoModelForQuestionAnswering
        .from_pretrained(MODEL_NAME)
    )

    write_phase(
        phase_file,
        "loading_dataset",
    )

    dataset = load_dataset(
        "rajpurkar/squad"
    )

    train_examples = dataset["train"]
    validation_examples = (
        dataset["validation"]
    )

    if args.max_train_samples is not None:
        n = min(
            args.max_train_samples,
            len(train_examples),
        )

        train_examples = (
            train_examples.select(range(n))
        )

    if args.max_eval_samples is not None:
        n = min(
            args.max_eval_samples,
            len(validation_examples),
        )

        validation_examples = (
            validation_examples.select(
                range(n)
            )
        )

    print(
        "Train examples: "
        f"{len(train_examples)}"
    )

    print(
        "Validation examples: "
        f"{len(validation_examples)}"
    )

    write_phase(
        phase_file,
        "tokenization",
    )

    tokenized_train = (
        train_examples.map(
            lambda examples:
                preprocess_examples(
                    examples,
                    tokenizer,
                ),
            batched=True,
            remove_columns=(
                train_examples.column_names
            ),
            desc="Tokenizing train",
        )
    )

    tokenized_validation = (
        validation_examples.map(
            lambda examples:
                preprocess_examples(
                    examples,
                    tokenizer,
                ),
            batched=True,
            remove_columns=(
                validation_examples
                .column_names
            ),
            desc="Tokenizing validation loss",
        )
    )

    validation_prediction_features = (
        validation_examples.map(
            lambda examples:
                prepare_validation_features(
                    examples,
                    tokenizer,
                ),
            batched=True,
            remove_columns=(
                validation_examples
                .column_names
            ),
            desc="Preparing validation QA metrics",
        )
    )

    prediction_model_dataset = (
        validation_prediction_features
        .remove_columns(
            [
                "example_id",
                "offset_mapping",
            ]
        )
    )

    train_features = len(
        tokenized_train
    )

    validation_features = len(
        tokenized_validation
    )

    print(
        "Train tokenized features: "
        f"{train_features}"
    )

    print(
        "Validation tokenized features: "
        f"{validation_features}"
    )

    full_batches = (
        train_features
        // args.batch_size
    )

    tail_batch_size = (
        train_features
        % args.batch_size
    )

    if args.drop_last:
        batches_per_epoch = (
            full_batches
        )

        features_per_epoch = (
            full_batches
            * args.batch_size
        )
    else:
        batches_per_epoch = math.ceil(
            train_features
            / args.batch_size
        )

        features_per_epoch = (
            train_features
        )

    dropped_per_epoch = (
        train_features
        - features_per_epoch
    )

    print("\n=== Batch geometry ===")
    print(
        f"Full train batches: "
        f"{full_batches}"
    )
    print(
        f"Train tail batch: "
        f"{tail_batch_size}"
    )
    print(
        "Train batches/epoch: "
        f"{batches_per_epoch}"
    )
    print(
        "Dropped train features/epoch: "
        f"{dropped_per_epoch}"
    )

    data_collator = (
        DefaultDataCollator()
    )

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),

        num_train_epochs=args.epochs,

        per_device_train_batch_size=(
            args.batch_size
        ),

        per_device_eval_batch_size=(
            args.eval_batch_size
        ),

        learning_rate=(
            args.learning_rate
        ),

        weight_decay=0.01,

        optim="adamw_torch_fused",

        lr_scheduler_type="linear",

        warmup_steps=args.warmup_steps,

        gradient_accumulation_steps=1,

        max_grad_norm=1.0,

        eval_strategy="epoch",

        save_strategy="epoch",

        load_best_model_at_end=True,

        metric_for_best_model=(
            "eval_loss"
        ),

        greater_is_better=False,

        save_total_limit=2,

        # Saves model weights but not the large
        # optimizer/scheduler state.
        save_only_model=True,

        logging_strategy="steps",

        logging_steps=50,

        logging_first_step=True,

        report_to=["tensorboard"],

        run_name=(
            "bert-squad-validation"
        ),

        seed=args.seed,
        data_seed=args.seed,

        dataloader_num_workers=(
            args.dataloader_num_workers
        ),

        dataloader_pin_memory=True,

        # Custom Trainer below forces this
        # to False during eval/test.
        dataloader_drop_last=(
            args.drop_last
        ),

        label_names=[
            "start_positions",
            "end_positions",
        ],

        bf16=args.bf16,

        torch_compile=(
            args.torch_compile
        ),

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

        skip_memory_metrics=True,
    )

    telemetry = ResourceTimingCallback(
        output_dir
    )

    trainer = InstrumentedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_validation,
        processing_class=tokenizer,
        data_collator=data_collator,
        phase_file=str(phase_file),
        callbacks=[
            telemetry,
            EarlyStoppingCallback(
                early_stopping_patience=(
                    args.early_stopping_patience
                ),
                early_stopping_threshold=(
                    args.early_stopping_threshold
                ),
            ),
        ],
    )

    print("\n=== Resolved Trainer config ===")
    print(
        f"Optimizer: "
        f"{training_args.optim}"
    )
    print(
        "Scheduler: "
        f"{training_args.lr_scheduler_type}"
    )
    print(
        "load_best_model_at_end: "
        f"{training_args.load_best_model_at_end}"
    )
    print(
        "metric_for_best_model: "
        f"{training_args.metric_for_best_model}"
    )

    print("\n=== Starting training ===")

    write_phase(
        phase_file,
        "training",
    )

    torch.cuda.synchronize()

    training_start = (
        time.perf_counter()
    )

    train_result = trainer.train()

    torch.cuda.synchronize()

    training_end = (
        time.perf_counter()
    )

    train_call_wall_seconds = (
        training_end
        - training_start
    )

    eval_wall_during_training = (
        trainer.eval_wall_seconds
    )

    train_log_history = list(
        trainer.state.log_history
    )

    eval_entries = [
        entry
        for entry in train_log_history
        if "eval_loss" in entry
    ]

    best_epoch = None

    if eval_entries:
        best_entry = min(
            eval_entries,
            key=lambda x: x[
                "eval_loss"
            ],
        )

        best_epoch = (
            best_entry.get("epoch")
        )

    print("\n=== Training complete ===")
    print(
        "Training call wall time: "
        f"{train_call_wall_seconds:.2f} s"
    )
    print(
        "Evaluation wall during train: "
        f"{eval_wall_during_training:.2f} s"
    )
    print(
        "Training-step callback wall: "
        f"{telemetry.training_step_wall_seconds:.2f} s"
    )
    print(
        f"Completed epoch: "
        f"{trainer.state.epoch}"
    )
    print(
        f"Global step: "
        f"{trainer.state.global_step}"
    )
    print(
        f"Best eval loss: "
        f"{trainer.state.best_metric}"
    )
    print(
        f"Best epoch: "
        f"{best_epoch}"
    )
    print(
        "Best checkpoint: "
        f"{trainer.state.best_model_checkpoint}"
    )

    write_phase(
        phase_file,
        "final_best_validation",
    )

    trainer.set_eval_phase(
        "final_best_validation",
        "post_training",
    )

    final_best_eval_metrics = (
        trainer.evaluate()
    )

    print("\n=== SQuAD EM/F1 on best model ===")

    prediction_start = (
        time.perf_counter()
    )

    prediction_output = trainer.predict(
        prediction_model_dataset
    )

    prediction_wall_seconds = (
        time.perf_counter()
        - prediction_start
    )

    predictions = (
        postprocess_qa_predictions(
            validation_examples,
            validation_prediction_features,
            prediction_output.predictions,
        )
    )

    squad_metrics = (
        compute_squad_metrics(
            validation_examples,
            predictions,
        )
    )

    print(
        "Exact Match: "
        f"{squad_metrics['exact_match']:.2f}"
    )

    print(
        "F1: "
        f"{squad_metrics['f1']:.2f}"
    )

    final_model_dir = output_dir / "best_model"

    trainer.save_model(
        str(final_model_dir)
    )

    tokenizer.save_pretrained(
        str(final_model_dir)
    )

    print(
        "Best model saved permanently to: "
        f"{final_model_dir}"
    )

    best_checkpoint = (
        trainer.state.best_model_checkpoint
    )

    if best_checkpoint:
        tokenizer.save_pretrained(
            best_checkpoint
        )

        (
            output_dir
            / "BEST_CHECKPOINT.txt"
        ).write_text(
            best_checkpoint + "\n",
            encoding="utf-8",
        )

    with (
        output_dir
        / "trainer_log_history.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            trainer.state.log_history,
            f,
            indent=4,
            default=str,
        )

    early_stopped = (
        trainer.state.epoch is not None
        and trainer.state.epoch
        < args.epochs
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

        "max_epochs": args.epochs,

        "completed_epoch": (
            trainer.state.epoch
        ),

        "early_stopped": (
            early_stopped
        ),

        "early_stopping_patience": (
            args.early_stopping_patience
        ),

        "early_stopping_threshold": (
            args.early_stopping_threshold
        ),

        "best_epoch": best_epoch,

        "best_eval_loss": (
            trainer.state.best_metric
        ),

        "best_model_checkpoint": (
            best_checkpoint
        ),

        "global_step": (
            trainer.state.global_step
        ),

        "batch_size": (
            args.batch_size
        ),

        "eval_batch_size": (
            args.eval_batch_size
        ),

        "learning_rate": (
            args.learning_rate
        ),

        "bf16": args.bf16,

        "drop_last_training": (
            args.drop_last
        ),

        "drop_last_validation": False,

        "torch_compile": (
            args.torch_compile
        ),

        "torch_compile_mode": (
            args.torch_compile_mode
            if args.torch_compile
            else None
        ),

        "optimizer": str(
            training_args.optim
        ),

        "lr_scheduler": str(
            training_args.lr_scheduler_type
        ),

        "warmup_steps": args.warmup_steps,

        "train_examples": (
            len(train_examples)
        ),

        "train_features": (
            train_features
        ),

        "validation_examples": (
            len(validation_examples)
        ),

        "validation_features": (
            validation_features
        ),

        "train_batches_per_epoch": (
            batches_per_epoch
        ),

        "train_tail_batch_size": (
            tail_batch_size
        ),

        "dropped_train_features_per_epoch": (
            dropped_per_epoch
        ),

        "train_call_wall_seconds": (
            train_call_wall_seconds
        ),

        "eval_wall_during_training_seconds": (
            eval_wall_during_training
        ),

        "non_eval_train_call_wall_seconds": (
            train_call_wall_seconds
            - eval_wall_during_training
        ),

        "training_step_callback_wall_seconds": (
            telemetry.training_step_wall_seconds
        ),

        "max_gpu_allocated_gib_seen": (
            telemetry.max_allocated_gib_seen
        ),

        "max_gpu_reserved_gib_seen": (
            telemetry.max_reserved_gib_seen
        ),

        "trainer_train_metrics": (
            train_result.metrics
        ),

        "final_best_eval_metrics": (
            final_best_eval_metrics
        ),

        "squad_exact_match": (
            squad_metrics["exact_match"]
        ),

        "squad_f1": (
            squad_metrics["f1"]
        ),

        "squad_prediction_wall_seconds": (
            prediction_wall_seconds
        ),

        "prediction_trainer_metrics": (
            prediction_output.metrics
        ),
    }

    results_path = (
        output_dir
        / "experiment_results.json"
    )

    with results_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=4,
            default=str,
        )

    write_phase(
        phase_file,
        "finished",
    )

    print(
        "\nResults written to: "
        f"{results_path}"
    )


if __name__ == "__main__":
    main()
