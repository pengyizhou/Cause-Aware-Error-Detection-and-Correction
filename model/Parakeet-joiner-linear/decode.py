"""
Decoding script for Parakeet-Joint-Linear frame-level classifiers.

This script loads a trained classifier checkpoint (operating on 640‑dim
joint embeddings) and runs inference on:

- A HuggingFace parquet dataset of embeddings, **or**
- A `joint_embeddings.pt` file produced by `Parakeet-ASR/inference_with_joint_emb.py`.

For evaluation on Kaldi-style test sets you will typically:
1) Extract joint embeddings with `inference_with_joint_emb.py` (to get `joint_embeddings.pt`)
2) Run this script with `--embeddings_pt joint_embeddings.pt`
3) Use the `.labels.txt` output files for downstream analysis / fusion.
"""

import os
import argparse
import json
from typing import List, Optional, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from model import ParakeetForDistortionDetection, load_model_for_training
from dataset_hf_parquet import HFParquetDistortionDataset, collate_fn_hf_parquet


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class JointEmbeddingsDictDataset(Dataset):
    """
    Simple Dataset that reads embeddings from a `joint_embeddings.pt` file.

    Expected format of the .pt file (as produced by `inference_with_joint_emb.py`):

        {
            "uttid1": Tensor[time, 640],
            "uttid2": Tensor[time, 640],
            ...
        }
    """

    def __init__(
        self,
        embeddings_path: str,
        max_length_frames: Optional[int] = None,
    ):
        super().__init__()
        if not os.path.exists(embeddings_path):
            raise ValueError(f"Embeddings file not found: {embeddings_path}")

        print(f"Loading joint embeddings from {embeddings_path} ...")
        raw_dict: Dict[str, Any] = torch.load(embeddings_path, map_location="cpu")

        self.uttids: List[str] = sorted(raw_dict.keys())
        self.embeddings: Dict[str, torch.Tensor] = {}
        self.max_length_frames = max_length_frames

        for uttid in self.uttids:
            emb = raw_dict[uttid]
            if isinstance(emb, np.ndarray):
                emb = torch.from_numpy(emb)
            elif not isinstance(emb, torch.Tensor):
                emb = torch.tensor(emb)

            emb = emb.float()  # [time, dim]

            if emb.dim() != 2:
                raise ValueError(
                    f"Expected 2-D embedding for {uttid}, got shape {tuple(emb.shape)}"
                )

            if max_length_frames is not None and emb.size(0) > max_length_frames:
                # Simple truncation for very long utterances
                emb = emb[:max_length_frames, :]

            self.embeddings[uttid] = emb

        print(f"  Loaded {len(self.uttids)} utterances from embeddings dict.")

    def __len__(self) -> int:
        return len(self.uttids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        uttid = self.uttids[idx]
        emb = self.embeddings[uttid]
        time = emb.size(0)

        # Attention mask: all ones for valid frames
        attention_mask = torch.ones(time, dtype=torch.long)

        return {
            "uttid": uttid,
            "embedding": emb,  # [time, dim]
            "attention_mask": attention_mask,
        }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: str = "cuda",
    input_dim: int = 640,
    num_labels: int = 2,
    dropout_prob: float = 0.2,
    hidden_dim: Optional[int] = None,
    cnn_kernel_size: Optional[int] = None,
    cnn_num_layers: int = 2,
    conv_transformer: bool = False,
    conv_transformer_num_heads: int = 8,
    conv_transformer_num_layers: int = 3,
) -> ParakeetForDistortionDetection:
    """
    Load a Parakeet-Joint-Linear classifier from a checkpoint directory.

    The checkpoint directory is expected to contain at least:
      - classifier.pt
      - cnn.pt (if CNN / Conv-Transformer was used)
      - config.json (created by the training script)
    """
    print(f"Loading joint-linear model from {checkpoint_path} ...")

    if not os.path.isdir(checkpoint_path):
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint_path}")

    config_path = os.path.join(checkpoint_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        # Override defaults with config where available
        num_labels = config.get("num_labels", num_labels)
        input_dim = config.get("input_dim", input_dim)
        hidden_dim = config.get("hidden_dim", hidden_dim)
        use_cnn = config.get("use_cnn", False)
        use_conv_transformer = config.get("use_conv_transformer", False)
        print(
            f"  Loaded config: num_labels={num_labels}, "
            f"input_dim={input_dim}, hidden_dim={hidden_dim}, "
            f"use_cnn={use_cnn}, use_conv_transformer={use_conv_transformer}"
        )
        if not use_cnn:
            cnn_kernel_size = None
    else:
        use_cnn = cnn_kernel_size is not None
        use_conv_transformer = conv_transformer

    # Build model with architecture matching training
    model = load_model_for_training(
        input_dim=input_dim,
        num_labels=num_labels,
        dropout_prob=dropout_prob,
        hidden_dim=hidden_dim,
        cnn_kernel_size=cnn_kernel_size,
        cnn_num_layers=cnn_num_layers,
        conv_transformer=use_conv_transformer,
        conv_transformer_num_heads=conv_transformer_num_heads,
        conv_transformer_num_layers=conv_transformer_num_layers,
    )

    # Load weights
    classifier_path = os.path.join(checkpoint_path, "classifier.pt")
    cnn_path = os.path.join(checkpoint_path, "cnn.pt")
    conv_layers_path = os.path.join(checkpoint_path, "conv_layers.pt")
    transformer_path = os.path.join(checkpoint_path, "transformer_encoder.pt")

    if os.path.exists(classifier_path):
        print(f"  Loading classifier from {classifier_path} ...")
        state = torch.load(classifier_path, map_location="cpu")
        model.classifier.load_state_dict(state)
        print("  ✓ Loaded classifier weights")
    else:
        raise ValueError(f"Classifier checkpoint not found at {classifier_path}")

    if os.path.exists(cnn_path) and hasattr(model, "cnn"):
        print(f"  Loading CNN from {cnn_path} ...")
        state = torch.load(cnn_path, map_location="cpu")
        model.cnn.load_state_dict(state)
        print("  ✓ Loaded CNN weights")

    if os.path.exists(conv_layers_path) and hasattr(model, "conv_layers"):
        print(f"  Loading Conv layers from {conv_layers_path} ...")
        state = torch.load(conv_layers_path, map_location="cpu")
        model.conv_layers.load_state_dict(state)
        print("  ✓ Loaded Conv layers weights")

    if os.path.exists(transformer_path) and hasattr(model, "transformer_encoder"):
        print(f"  Loading Transformer encoder from {transformer_path} ...")
        state = torch.load(transformer_path, map_location="cpu")
        model.transformer_encoder.load_state_dict(state)
        print("  ✓ Loaded Transformer encoder weights")

    model = model.to(device)
    model.eval()
    print(f"  Model loaded successfully on {device}")
    return model


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def decode(
    model: ParakeetForDistortionDetection,
    dataloader: DataLoader,
    device: str,
    output_file: str,
    label_names: List[str],
    output_format: str = "per_frame",
    compute_metrics: bool = True,
):
    """
    Run decoding, write per-utterance results to JSONL, and metrics to TXT.

    Args:
        model: Trained joint-linear model
        dataloader: DataLoader for test data
        device: Device to run inference on
        output_file: Path to output text file
        label_names: Names for each label class
        output_format: "per_frame", "summary", or "both"
        compute_metrics: Whether to compute metrics (only if labels are available)
    """
    model.eval()

    all_predictions: List[int] = []
    all_labels: List[int] = []
    all_results: List[Dict[str, Any]] = []

    print("\nRunning joint-linear inference...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Decoding"):
            if batch is None:
                continue

            embeddings = batch["embedding"].to(device)  # [B, T, D]
            attention_mask = batch["attention_mask"].to(device)
            uttids = batch["uttid"]
            embedding_lengths = batch.get("embedding_lengths", None)

            has_labels = "labels" in batch
            if has_labels:
                labels = batch["labels"].to(device)
                label_lengths = batch.get("label_lengths", None)

            outputs = model(input_values=embeddings, labels=labels if has_labels else None)
            logits = outputs["logits"]  # [B, T, C]
            predictions = torch.argmax(logits, dim=-1)  # [B, T]

            batch_size = embeddings.size(0)

            for i in range(batch_size):
                uttid = uttids[i]

                if embedding_lengths is not None:
                    num_frames = int(embedding_lengths[i])
                else:
                    # Fall back to full sequence length
                    num_frames = int(predictions.size(1))

                sample_preds = predictions[i, :num_frames].cpu().numpy()

                result: Dict[str, Any] = {
                    "file_id": uttid,
                    "predictions": sample_preds,
                    "num_frames": num_frames,
                }

                if has_labels:
                    # Align with reference labels if provided
                    sample_labels = labels[i]
                    if label_lengths is not None:
                        sample_labels = sample_labels[: int(label_lengths[i])]

                    sample_labels = sample_labels.cpu().numpy()
                    min_len = min(len(sample_preds), len(sample_labels))
                    sample_preds_aligned = sample_preds[:min_len]
                    sample_labels_aligned = sample_labels[:min_len]

                    mask = sample_labels_aligned != -100
                    sample_preds_aligned = sample_preds_aligned[mask]
                    sample_labels_aligned = sample_labels_aligned[mask]

                    result["labels"] = sample_labels

                    all_predictions.extend(sample_preds_aligned.tolist())
                    all_labels.extend(sample_labels_aligned.tolist())

                all_results.append(result)

    # ------------------------------------------------------------------
    # Write detailed per-utterance results to JSONL (one JSON per line)
    # ------------------------------------------------------------------
    base, ext = os.path.splitext(output_file)
    jsonl_path = base + ".jsonl" if ext else output_file + ".jsonl"

    jsonl_dir = os.path.dirname(jsonl_path)
    if jsonl_dir:
        os.makedirs(jsonl_dir, exist_ok=True)

    print(f"\nWriting per-utterance results to {jsonl_path} ...")
    with open(jsonl_path, "w") as jf:
        for result in all_results:
            file_id = result["file_id"]
            predictions = result["predictions"]
            num_frames = int(result["num_frames"])

            # Convert to plain Python types for JSON serialization
            if isinstance(predictions, np.ndarray):
                pred_list = predictions.tolist()
            else:
                pred_list = list(predictions)

            entry: Dict[str, Any] = {
                "file_id": file_id,
                "num_frames": num_frames,
                "predictions": [int(p) for p in pred_list],
            }

            labels_arr = result.get("labels")
            if labels_arr is not None:
                if isinstance(labels_arr, np.ndarray):
                    labels_list = labels_arr.tolist()
                else:
                    labels_list = list(labels_arr)
                entry["labels"] = [int(l) for l in labels_list]

            json.dump(entry, jf)
            jf.write("\n")

    print(f"  ✓ Wrote {len(all_results)} utterance results to {jsonl_path}")

    # Companion labels file with human-readable labels per frame
    labels_file = output_file.replace(".txt", ".labels.txt")
    with open(labels_file, "w") as f:
        for result in all_results:
            file_id = result["file_id"]
            predictions = result["predictions"]
            pred_labels = [
                label_names[p] if p < len(label_names) else f"Class_{p}"
                for p in predictions
            ]
            pred_str = " ".join(pred_labels)
            f.write(f"{file_id}\t{pred_str}\n")
    print(f"  ✓ Wrote label names to {labels_file}")

    # Metrics (if labels available)
    if compute_metrics and len(all_labels) > 0:
        print("\n" + "=" * 60)
        print("Evaluation Metrics")
        print("=" * 60)

        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average="weighted", zero_division=0
        )
        report_str = classification_report(
            all_labels,
            all_predictions,
            labels=list(range(len(label_names))),
            target_names=label_names,
            zero_division=0,
        )
        cm = confusion_matrix(
            all_labels, all_predictions, labels=list(range(len(label_names)))
        )

        # Print metrics to stdout (unchanged behaviour)
        print("\nOverall Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")

        print("\nPer-class Metrics:")
        print(report_str)

        print("\nConfusion Matrix:")
        print(f"{'':15s}", end="")
        for name in label_names:
            print(f"{name[:10]:>12s}", end="")
        print()
        for i, name in enumerate(label_names):
            print(f"{name:15s}", end="")
            for j in range(len(label_names)):
                print(f"{cm[i, j]:12d}", end="")
            print()

        # JSON metrics file (as before)
        metrics_file = output_file.replace(".txt", ".metrics.json")
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm.tolist(),
            "label_names": label_names,
            "num_samples": len(all_results),
            "num_frames": len(all_labels),
        }
        metrics_dir = os.path.dirname(metrics_file)
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  ✓ Saved metrics to {metrics_file}")

        # Human-readable metrics TXT (no per-utterance predictions)
        metrics_txt = output_file
        metrics_txt_dir = os.path.dirname(metrics_txt)
        if metrics_txt_dir:
            os.makedirs(metrics_txt_dir, exist_ok=True)

        with open(metrics_txt, "w") as f:
            f.write("Parakeet Joint-Linear Distortion Detection Metrics\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Num samples: {len(all_results)}\n")
            f.write(f"Num labeled frames: {len(all_labels)}\n\n")
            f.write("Overall:\n")
            f.write(f"  Accuracy:  {accuracy:.4f}\n")
            f.write(f"  Precision: {precision:.4f}\n")
            f.write(f"  Recall:    {recall:.4f}\n")
            f.write(f"  F1-Score:  {f1:.4f}\n\n")
            f.write("Per-class report:\n")
            f.write(report_str)
            f.write("\n\nConfusion matrix:\n")
            f.write(f"{'':15s}")
            for name in label_names:
                f.write(f"{name[:10]:>12s}")
            f.write("\n")
            for i, name in enumerate(label_names):
                f.write(f"{name:15s}")
                for j in range(len(label_names)):
                    f.write(f"{cm[i, j]:12d}")
                f.write("\n")
        print(f"  ✓ Saved metrics summary to {metrics_txt}")

    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Decode Parakeet-Joint-Linear distortion detection model"
    )

    # Required paths
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint directory (e.g., output/.../best_model)",
    )

    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--data_dir",
        type=str,
        help="Directory containing parquet files to decode (HF parquet embeddings)",
    )
    data_group.add_argument(
        "--embeddings_pt",
        type=str,
        help="Path to joint_embeddings.pt file (uttid -> [time, 640])",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to output text file",
    )

    # Model hyperparameters (must match training)
    parser.add_argument(
        "--input_dim",
        type=int,
        default=640,
        help="Input embedding dimension (default: 640)",
    )
    parser.add_argument(
        "--num_labels",
        type=int,
        default=2,
        help="Number of classification labels (default: 2)",
    )
    parser.add_argument(
        "--dropout_prob",
        type=float,
        default=0.2,
        help="Dropout probability (default: 0.2)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=None,
        help="Hidden dimension for 2-layer classifier (if used during training)",
    )
    parser.add_argument(
        "--cnn_kernel_size",
        type=int,
        default=5,
        help="Kernel size for CNN classifier (must match training; default: 5)",
    )
    parser.add_argument(
        "--cnn_num_layers",
        type=int,
        default=5,
        help="Number of CNN layers (must match training; default: 5)",
    )
    parser.add_argument(
        "--use_conv_transformer",
        action="store_true",
        help="Use Conv-Transformer architecture (must match training)",
    )
    parser.add_argument(
        "--conv_transformer_num_heads",
        type=int,
        default=8,
        help="Number of attention heads for Conv-Transformer (default: 8)",
    )
    parser.add_argument(
        "--conv_transformer_num_layers",
        type=int,
        default=3,
        help="Number of Transformer encoder layers for Conv-Transformer (default: 3)",
    )

    # Decoding parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for decoding (default: 32)",
    )
    parser.add_argument(
        "--max_length_frames",
        type=int,
        default=None,
        help="Optional maximum number of frames per utterance (for embeddings_pt mode)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)",
    )

    # Output formatting
    parser.add_argument(
        "--output_format",
        type=str,
        default="per_frame",
        choices=["per_frame", "summary", "both"],
        help="Output format: per_frame predictions, summary stats, or both",
    )
    parser.add_argument(
        "--label_names",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Names for each label class. If omitted, generic names "
            "Class_0, Class_1, ... are used."
        ),
    )
    parser.add_argument(
        "--no_metrics",
        action="store_true",
        help="Do not compute evaluation metrics (useful when no labels are available)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Label names
    if args.label_names is None:
        label_names = [f"Class_{i}" for i in range(args.num_labels)]
    else:
        label_names = args.label_names

    # Load model
    model = load_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        device=device,
        input_dim=args.input_dim,
        num_labels=args.num_labels,
        dropout_prob=args.dropout_prob,
        hidden_dim=args.hidden_dim,
        cnn_kernel_size=args.cnn_kernel_size,
        cnn_num_layers=args.cnn_num_layers,
        conv_transformer=args.use_conv_transformer,
        conv_transformer_num_heads=args.conv_transformer_num_heads,
        conv_transformer_num_layers=args.conv_transformer_num_layers,
    )

    # Build dataset + dataloader
    if args.embeddings_pt is not None:
        print(f"\nLoading embeddings from {args.embeddings_pt} ...")
        dataset = JointEmbeddingsDictDataset(
        embeddings_path=args.embeddings_pt,
        max_length_frames=args.max_length_frames,
        )
    else:
        print(f"\nLoading parquet embeddings from {args.data_dir} ...")
        dataset = HFParquetDistortionDataset(
            parquet_dir=args.data_dir,
            max_length_seconds=30.0,  # not used directly for embeddings
            shuffle=False,
            streaming=False,
            return_file_path=False,
            require_labels=False,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_hf_parquet,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Decode
    decode(
        model=model,
        dataloader=dataloader,
        device=device,
        output_file=args.output_file,
        label_names=label_names,
        output_format=args.output_format,
        compute_metrics=not args.no_metrics,
    )

    print("\n" + "=" * 60)
    print("Joint-linear decoding completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

