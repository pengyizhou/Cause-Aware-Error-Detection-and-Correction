"""
Decoding script for Parakeet-Encoder based frame-level distortion detection.

Loads a trained model checkpoint and runs inference on parquet data.
Writes classification results to text files.
"""

import os
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, List
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

from model import ParakeetForDistortionDetection
from dataset_hf_parquet import HFParquetDistortionDataset, collate_fn_hf_parquet


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: str = "cuda",
    num_labels: int = 6,
    projector_hidden_dim: Optional[int] = None,
    cnn_kernel_size: Optional[int] = None,
    cnn_num_layers: int = 2,
    use_encoder_projection: bool = False,
) -> ParakeetForDistortionDetection:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint directory (e.g., output/best_model)
        device: Device to load model on
        num_labels: Number of classification labels
        projector_hidden_dim: Hidden dim for 2-layer projector (None for single layer)
        
    Returns:
        Loaded model in eval mode
    """
    print(f"Loading model from {checkpoint_path}...")
    
    # Load config if available
    config_path = os.path.join(checkpoint_path, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        num_labels = config.get('num_labels', num_labels)
        print(f"  Loaded config: num_labels={num_labels}")
    
    # Build model
    model = ParakeetForDistortionDetection(
        parakeet_model_name="nvidia/parakeet-tdt-0.6b-v2",
        num_labels=num_labels,
        freeze_encoder=True,
        freeze_preprocessor=True,
        projector_hidden_dim=projector_hidden_dim,
        cnn_kernel_size=cnn_kernel_size,
        cnn_num_layers=cnn_num_layers,
        use_encoder_projection=use_encoder_projection,
    )
    
    # Load checkpoint weights
    parakeet_path = os.path.join(checkpoint_path, 'parakeet_modules.pt')
    classifier_path = os.path.join(checkpoint_path, 'classifier.pt')
    cnn_path = os.path.join(checkpoint_path, 'cnn.pt')
    
    if os.path.exists(parakeet_path):
        print(f"  Loading Parakeet modules from {parakeet_path}...")
        parakeet_state = torch.load(parakeet_path, map_location='cpu')
        model.preprocessor.load_state_dict(parakeet_state['preprocessor'])
        model.encoder.load_state_dict(parakeet_state['encoder'])
        print("  ✓ Loaded Parakeet preprocessor and encoder")
    
    if os.path.exists(classifier_path):
        print(f"  Loading classifier from {classifier_path}...")
        model.classifier.load_state_dict(torch.load(classifier_path, map_location='cpu'))
        print("  ✓ Loaded classifier")
    else:
        raise ValueError(f"Classifier checkpoint not found at {classifier_path}")
    
    if os.path.exists(cnn_path) and cnn_kernel_size is not None:
        print(f"  Loading CNN from {cnn_path}...")
        cnn_state = torch.load(cnn_path, map_location='cpu')
        model.cnn.load_state_dict(cnn_state)
        print("  ✓ Loaded CNN")
    else:
        raise ValueError(f"CNN checkpoint not found at {cnn_path}")
    
    
    model = model.to(device)
    model.eval()
    
    print(f"  Model loaded successfully on {device}")
    return model


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
        model: Trained model
        dataloader: DataLoader for test data
        device: Device to run inference on
        output_file: Path to output text file
        label_names: Names for each label class
        output_format: Output format - "per_frame", "summary", or "both"
        compute_metrics: Whether to compute and print metrics (requires labels)
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_results = []
    
    print(f"\nRunning inference...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Decoding"):
            input_values = batch['input_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            file_ids = batch['uttid']
            audio_lengths = batch['audio_lengths']
            
            has_labels = 'labels' in batch
            if has_labels:
                labels = batch['labels'].to(device)
                label_lengths = batch['label_lengths']
            
            # Forward pass
            outputs = model(
                input_values=input_values,
                attention_mask=attention_mask,
            )
            
            logits = outputs['logits']  # [batch, num_frames, num_labels]
            encoder_lengths = outputs['encoder_lengths']
            
            # Get predictions
            predictions = torch.argmax(logits, dim=-1)  # [batch, num_frames]
            
            # Process each sample in batch
            for i in range(len(file_ids)):
                file_id = file_ids[i]
                num_frames = encoder_lengths[i].item()
                
                # Get predictions for this sample (only valid frames)
                sample_preds = predictions[i, :num_frames].cpu().numpy()
                
                result = {
                    'file_id': file_id,
                    'predictions': sample_preds,
                    'num_frames': num_frames,
                }
                
                # Get reference labels if available
                if has_labels:
                    sample_labels = labels[i, :label_lengths[i]].cpu().numpy()
                    
                    # Align lengths
                    min_len = min(len(sample_preds), len(sample_labels))
                    sample_preds_aligned = sample_preds[:min_len]
                    sample_labels_aligned = sample_labels[:min_len]
                    
                    # Filter out -100
                    mask = sample_labels_aligned != -100
                    sample_preds_aligned = sample_preds_aligned[mask]
                    sample_labels_aligned = sample_labels_aligned[mask]
                    
                    result['labels'] = sample_labels
                    
                    all_predictions.extend(sample_preds_aligned)
                    all_labels.extend(sample_labels_aligned)
                
                all_results.append(result)
    
    # ------------------------------------------------------------------
    # Write detailed per-utterance results to JSONL (one JSON per line)
    # ------------------------------------------------------------------
    base, ext = os.path.splitext(output_file)
    jsonl_path = base + ".jsonl" if ext else output_file + ".jsonl"

    jsonl_dir = os.path.dirname(jsonl_path)
    if jsonl_dir:
        os.makedirs(jsonl_dir, exist_ok=True)

    print(f"\nWriting per-utterance results to {jsonl_path}...")
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

            entry: dict = {
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

    # Write per-frame predictions in a separate file (label format)
    labels_file = output_file.replace('.txt', '.labels.txt')
    with open(labels_file, 'w') as f:
        for result in all_results:
            file_id = result['file_id']
            predictions = result['predictions']
            # Convert to label names
            pred_labels = [label_names[p] if p < len(label_names) else f"Class_{p}" for p in predictions]
            pred_str = " ".join(pred_labels)
            f.write(f"{file_id}\t{pred_str}\n")
    print(f"  ✓ Wrote label names to {labels_file}")
    
    # Compute and print metrics if labels available
    if compute_metrics and len(all_labels) > 0:
        print(f"\n{'='*60}")
        print("Evaluation Metrics")
        print(f"{'='*60}")
        
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
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")

        print(f"\nPer-class Metrics:")
        print(report_str)

        print(f"\nConfusion Matrix:")
        print(f"{'':15s}", end="")
        for name in label_names:
            print(f"{name[:10]:>12s}", end="")
        print()
        for i, name in enumerate(label_names):
            print(f"{name:15s}", end="")
            for j in range(len(label_names)):
                print(f"{cm[i, j]:12d}", end="")
            print()

        # Save metrics to JSON
        metrics_file = output_file.replace('.txt', '.metrics.json')
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm.tolist(),
            'label_names': label_names,
            'num_samples': len(all_results),
            'num_frames': len(all_labels),
        }
        metrics_dir = os.path.dirname(metrics_file)
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  ✓ Saved metrics to {metrics_file}")

        # Human-readable metrics TXT (no per-utterance predictions)
        metrics_txt = output_file
        metrics_txt_dir = os.path.dirname(metrics_txt)
        if metrics_txt_dir:
            os.makedirs(metrics_txt_dir, exist_ok=True)

        with open(metrics_txt, "w") as f:
            f.write("Parakeet Encoder-Linear Distortion Detection Metrics\n")
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


def main():
    parser = argparse.ArgumentParser(description="Decode distortion detection model")
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint directory')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing parquet files to decode')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Path to output text file')
    
    # Model arguments
    parser.add_argument('--num_labels', type=int, default=2,
                       help='Number of classification labels')
    parser.add_argument('--deletion_only', action='store_true', default=False,
                       help='Only decode deletion errors')
    parser.add_argument('--cnn_kernel_size', type=int, default=None,
                       help='Kernel size for CNN classifier (None for linear classifier)')
    parser.add_argument('--cnn_num_layers', type=int, default=2,
                       help='Number of CNN layers (default: 2)')
    parser.add_argument('--projector_hidden_dim', type=int, default=None,
                       help='Hidden dim for 2-layer projector (None for single layer)')
    parser.add_argument('--use_encoder_projection', action='store_true',
                       help='Use the encoder linear projector from Parakeet joint network (e.g., 1024 -> 640) before classifier')

    
    # Decoding arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for decoding')
    parser.add_argument('--max_length_seconds', type=float, default=30.0,
                       help='Maximum audio length in seconds')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Output arguments
    parser.add_argument('--output_format', type=str, default='per_frame',
                       choices=['per_frame', 'summary', 'both'],
                       help='Output format: per_frame predictions, summary stats, or both')
    parser.add_argument('--label_names', type=str, nargs='+',
                       default=['Clean', 'Noisy', 'RIR', 'Interference', 'PacketLoss', 'Missing'],
                       help='Names for each label class')
    parser.add_argument('--no_metrics', action='store_true',
                       help='Do not compute evaluation metrics')
    
    
    args = parser.parse_args()
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    model = load_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        device=device,
        num_labels=args.num_labels,
        projector_hidden_dim=args.projector_hidden_dim,
        cnn_kernel_size=args.cnn_kernel_size,
        cnn_num_layers=args.cnn_num_layers,
        use_encoder_projection=args.use_encoder_projection,
    )
    
    # Create dataset and dataloader
    print(f"\nLoading data from {args.data_dir}...")
    dataset = HFParquetDistortionDataset(
        parquet_dir=args.data_dir,
        feature_extractor=None,  # Parakeet handles preprocessing internally
        max_length_seconds=args.max_length_seconds,
        sample_rate=16000,
        shuffle=False,  # No shuffle for decoding
        return_file_path=True,  # Need file paths for output
        require_labels=False,  # Labels are optional for inference
        remove_substitutions=True if args.deletion_only else False,
        remove_insertions=True if args.deletion_only else False
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_hf_parquet,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Run decoding
    results = decode(
        model=model,
        dataloader=dataloader,
        device=device,
        output_file=args.output_file,
        label_names=args.label_names,
        output_format=args.output_format,
        compute_metrics=not args.no_metrics,
    )
    
    print(f"\n{'='*60}")
    print("Decoding completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
