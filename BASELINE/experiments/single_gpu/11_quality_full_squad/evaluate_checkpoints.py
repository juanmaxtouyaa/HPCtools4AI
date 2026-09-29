import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)

import sys

BASELINE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASELINE_ROOT))

from train_v3_validation import (
    MODEL_NAME,
    prepare_validation_features,
    postprocess_qa_predictions,
    compute_squad_metrics,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint-dir",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=160,
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Expected 1 visible GPU, got "
            f"{torch.cuda.device_count()}"
        )

    checkpoint_root = Path(
        args.checkpoint_dir
    )

    output_root = Path(
        args.output_dir
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoints = sorted(
        checkpoint_root.glob("checkpoint-*"),
        key=lambda p: int(
            p.name.split("-")[-1]
        ),
    )

    if not checkpoints:
        raise RuntimeError(
            f"No checkpoints found in "
            f"{checkpoint_root}"
        )

    print("Checkpoints:")
    for checkpoint in checkpoints:
        print(" ", checkpoint)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    validation_examples = load_dataset(
        "rajpurkar/squad"
    )["validation"]

    validation_features = (
        validation_examples.map(
            lambda examples:
                prepare_validation_features(
                    examples,
                    tokenizer,
                ),
            batched=True,
            remove_columns=(
                validation_examples.column_names
            ),
            desc="Preparing SQuAD validation",
        )
    )

    prediction_dataset = (
        validation_features.remove_columns(
            [
                "example_id",
                "offset_mapping",
            ]
        )
    )

    results = []

    for epoch_index, checkpoint in enumerate(
        checkpoints,
        start=1,
    ):
        print()
        print(
            f"===== CHECKPOINT {checkpoint.name} ====="
        )

        model = (
            AutoModelForQuestionAnswering
            .from_pretrained(checkpoint)
        )

        args_eval = TrainingArguments(
            output_dir=str(
                output_root / "_eval_tmp"
            ),
            per_device_eval_batch_size=(
                args.eval_batch_size
            ),
            dataloader_num_workers=0,
            bf16=True,
            report_to=[],
        )

        trainer = Trainer(
            model=model,
            args=args_eval,
            processing_class=tokenizer,
            data_collator=DefaultDataCollator(),
        )

        output = trainer.predict(
            prediction_dataset
        )

        predictions = (
            postprocess_qa_predictions(
                validation_examples,
                validation_features,
                output.predictions,
            )
        )

        metrics = compute_squad_metrics(
            validation_examples,
            predictions,
        )

        record = {
            "epoch": epoch_index,
            "checkpoint": str(checkpoint),
            "global_step": int(
                checkpoint.name.split("-")[-1]
            ),
            "exact_match": (
                metrics["exact_match"]
            ),
            "f1": metrics["f1"],
            "prediction_metrics": (
                output.metrics
            ),
        }

        results.append(record)

        print(
            f"Epoch {epoch_index}: "
            f"EM={metrics['exact_match']:.2f} "
            f"F1={metrics['f1']:.2f}"
        )

        del trainer
        del model
        torch.cuda.empty_cache()

    best = max(
        results,
        key=lambda x: x["f1"],
    )

    print()
    print("===== BEST BY F1 =====")
    print(
        f"epoch={best['epoch']} "
        f"EM={best['exact_match']:.2f} "
        f"F1={best['f1']:.2f}"
    )

    best_model = (
        AutoModelForQuestionAnswering
        .from_pretrained(
            best["checkpoint"]
        )
    )

    best_dir = (
        output_root / "best_model_by_f1"
    )

    best_model.save_pretrained(
        best_dir,
        safe_serialization=True,
    )

    tokenizer.save_pretrained(
        best_dir
    )

    summary = {
        "results": results,
        "best_by_f1": best,
        "best_model_directory": (
            str(best_dir)
        ),
    }

    with (
        output_root
        / "per_epoch_squad_metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=4,
        )

    print(
        "Best-F1 model saved to:",
        best_dir,
    )


if __name__ == "__main__":
    main()
