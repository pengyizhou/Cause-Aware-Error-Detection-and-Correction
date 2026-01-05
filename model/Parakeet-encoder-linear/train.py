"""
Training script for Parakeet-Encoder based frame-level distortion detection.
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from model import load_model_for_training
from dataset_hf_parquet import HFParquetDistortionDataset, collate_fn_hf_parquet


class DistortionDetectionTrainer:
    """Trainer for Parakeet-based distortion detection model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        output_dir: str,
        num_epochs: int,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        log_interval: int = 10,
        eval_interval: int = 500,
        save_interval: int = 1000,
        num_labels: int = 6,
        local_rank: int = -1,
        world_size: int = 1,
        label_names: list = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = output_dir
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.num_labels = num_labels
        self.local_rank = local_rank
        self.world_size = world_size
        
        # Check if this is the main process
        self.is_main_process = (local_rank == -1 or local_rank == 0)
        
        # Create output directory (only on main process)
        if self.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
        
        # Initialize tensorboard (only on main process)
        if self.is_main_process:
            self.writer = SummaryWriter(os.path.join(output_dir, 'logs'))
        else:
            self.writer = None
        
        # Training state
        self.global_step = 0
        self.best_val_accuracy = 0.0
        
        # Label names
        self.label_names = label_names or [
            'Clean',
            'Noisy',
            'RIR',
            'Interference',
            'Packet Loss',
            'Missing',
            'RIR+Noise',
            'RIR+Interference'
        ]
    
    def train(self):
        """Main training loop."""
        if self.is_main_process:
            print(f"Starting training for {self.num_epochs} epochs...")
            print(f"Total training steps: {len(self.train_dataloader) * self.num_epochs}")
            print(f"Device: {self.device}")
            if self.world_size > 1:
                print(f"Distributed training with {self.world_size} GPUs")
        
        self.model.to(self.device)
        self.model.train()
        
        # When resume training, we need to skip already trained steps
        already_trained_steps = self.global_step * self.gradient_accumulation_steps
        batches_per_epoch = len(self.train_dataloader)
        
        for epoch in range(self.num_epochs):
            current_epoch_start = epoch * batches_per_epoch
            skip_batches = max(0, already_trained_steps - current_epoch_start)
            if self.is_main_process:
                print(f"\n{'='*50}")
                print(f"Epoch {epoch + 1}/{self.num_epochs}")
                print(f"{'='*50}")
            
            # Set epoch for distributed sampler
            if hasattr(self.train_dataloader.sampler, 'set_epoch'):
                self.train_dataloader.sampler.set_epoch(epoch)
            
            epoch_loss = 0.0
            self.optimizer.zero_grad()
            
            dataloader_iter = iter(self.train_dataloader)
            if skip_batches > 0 and skip_batches < batches_per_epoch:
                if self.is_main_process:
                    print(f"Skipping {skip_batches} already-trained batches in this epoch...")
                # Efficiently skip batches without loading them
                dataloader_iter = itertools.islice(dataloader_iter, skip_batches, None)
                batch_start_idx = skip_batches
            else:
                batch_start_idx = 0
            
            # Only show progress bar on main process
            if self.is_main_process:
                progress_bar = tqdm(
                    dataloader_iter, 
                    desc=f"Training",
                    initial=batch_start_idx,
                    total=batches_per_epoch
                )
            else:
                progress_bar = dataloader_iter
            
            for batch_idx, batch in enumerate(progress_bar):
                if batch is None:
                    continue
                
                # Move batch to device
                input_values = batch['input_values'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_values=input_values,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs['loss']
                loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                epoch_loss += loss.item()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                    
                    # Update weights
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                    
                    # Logging (only on main process)
                    if self.is_main_process and self.global_step % self.log_interval == 0:
                        lr = self.scheduler.get_last_lr()[0]
                        self.writer.add_scalar('train/loss', loss.item() * self.gradient_accumulation_steps, self.global_step)
                        self.writer.add_scalar('train/learning_rate', lr, self.global_step)
                    
                    # Evaluation
                    if self.global_step % self.eval_interval == 0:
                        val_metrics = self.evaluate()
                        if self.is_main_process:
                            self.log_metrics(val_metrics, self.global_step)
                        
                        # Save best model (only on main process)
                        if self.is_main_process and val_metrics['accuracy'] > self.best_val_accuracy:
                            self.best_val_accuracy = val_metrics['accuracy']
                            self.save_checkpoint('best_model')
                            print(f"\n✓ Saved best model (accuracy: {self.best_val_accuracy:.4f})")
                        
                        self.model.train()
                    
                    # Save checkpoint (only on main process)
                    if self.is_main_process and self.global_step % self.save_interval == 0:
                        self.save_checkpoint(f'checkpoint-{self.global_step}')
                
                # Update progress bar (only on main process)
                if self.is_main_process:
                    progress_bar.set_postfix({
                        'loss': loss.item() * self.gradient_accumulation_steps,
                        'lr': self.scheduler.get_last_lr()[0]
                    })
            
            # End of epoch
            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            if self.is_main_process:
                print(f"\nEpoch {epoch + 1} - Average Loss: {avg_epoch_loss:.4f}")
            
            # Evaluate at end of epoch
            val_metrics = self.evaluate()
            if self.is_main_process:
                self.log_metrics(val_metrics, self.global_step)
            
            # Save epoch checkpoint (only on main process)
            if self.is_main_process:
                self.save_checkpoint(f'epoch-{epoch + 1}')
        
        if self.is_main_process:
            print("\n" + "="*50)
            print("Training completed!")
            print(f"Best validation accuracy: {self.best_val_accuracy:.4f}")
            print("="*50)
        
        if self.writer is not None:
            self.writer.close()
    
    def evaluate(self):
        """Evaluate model on validation set."""
        if self.is_main_process:
            print("\n" + "-"*50)
            print("Evaluating...")
            print("-"*50)
        
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        
        with torch.no_grad():
            dataloader_iter = tqdm(self.val_dataloader, desc="Evaluation") if self.is_main_process else self.val_dataloader
            for batch in dataloader_iter:
                if batch is None:
                    continue
                
                input_values = batch['input_values'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_values=input_values,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_labels=True
                )
                
                loss = outputs['loss']
                logits = outputs['logits']
                labels_flat = outputs['labels']
                
                total_loss += loss.item()
                
                # Get predictions
                predictions = torch.argmax(logits, dim=-1)
                
                # Flatten and filter out padding (-100)
                predictions_flat = predictions.view(-1).cpu().numpy()
                labels_flat = labels_flat.cpu().numpy()
                
                mask = labels_flat != -100
                predictions_flat = predictions_flat[mask]
                labels_flat = labels_flat[mask]
                
                all_predictions.extend(predictions_flat)
                all_labels.extend(labels_flat)
        
        # Gather predictions and labels from all processes in DDP
        if self.world_size > 1:
            # Convert to tensors
            all_predictions_tensor = torch.tensor(all_predictions, dtype=torch.long, device=self.device)
            all_labels_tensor = torch.tensor(all_labels, dtype=torch.long, device=self.device)
            
            # Gather sizes from all processes
            local_size = torch.tensor([len(all_predictions)], dtype=torch.long, device=self.device)
            size_list = [torch.zeros_like(local_size) for _ in range(self.world_size)]
            dist.all_gather(size_list, local_size)
            
            # Gather predictions and labels
            max_size = max([s.item() for s in size_list])
            
            # Pad if necessary
            if len(all_predictions) < max_size:
                padding = torch.zeros(max_size - len(all_predictions), dtype=torch.long, device=self.device)
                all_predictions_tensor = torch.cat([all_predictions_tensor, padding])
                all_labels_tensor = torch.cat([all_labels_tensor, padding])
            
            # Gather from all processes
            gathered_predictions = [torch.zeros(max_size, dtype=torch.long, device=self.device) for _ in range(self.world_size)]
            gathered_labels = [torch.zeros(max_size, dtype=torch.long, device=self.device) for _ in range(self.world_size)]
            
            dist.all_gather(gathered_predictions, all_predictions_tensor)
            dist.all_gather(gathered_labels, all_labels_tensor)
            
            # Concatenate and trim to actual sizes
            all_predictions = []
            all_labels = []
            for i in range(self.world_size):
                size = size_list[i].item()
                all_predictions.extend(gathered_predictions[i][:size].cpu().numpy().tolist())
                all_labels.extend(gathered_labels[i][:size].cpu().numpy().tolist())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average='weighted',
            zero_division=0
        )
        
        # Per-class metrics
        per_class_precision, per_class_recall, per_class_f1, per_class_support = \
            precision_recall_fscore_support(
                all_labels,
                all_predictions,
                average=None,
                labels=list(range(self.num_labels)),
                zero_division=0
            )
        
        # Confusion matrix
        cm = confusion_matrix(
            all_labels,
            all_predictions,
            labels=list(range(self.num_labels))
        )
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'per_class_metrics': {
                'precision': per_class_precision,
                'recall': per_class_recall,
                'f1': per_class_f1,
                'support': per_class_support
            },
            'confusion_matrix': cm
        }
        
        # Print metrics (only on main process)
        if self.is_main_process:
            print(f"\nValidation Results:")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            
            print("\nPer-class metrics:")
            for i in range(self.num_labels):
                label_name = self.label_names[i] if i < len(self.label_names) else f"Class_{i}"
                print(f"  {label_name:15s} - "
                      f"P: {per_class_precision[i]:.3f}, "
                      f"R: {per_class_recall[i]:.3f}, "
                      f"F1: {per_class_f1[i]:.3f}, "
                      f"Support: {int(per_class_support[i])}")
            
            # Print confusion matrix
            print("\nConfusion Matrix (rows=actual, cols=predicted):")
            print(" " * 12, end="")
            for i in range(self.num_labels):
                label_name = self.label_names[i] if i < len(self.label_names) else f"C{i}"
                print(f"{label_name:>8s}", end="")
            print()
            for i in range(self.num_labels):
                label_name = self.label_names[i] if i < len(self.label_names) else f"C{i}"
                print(f"{label_name:12s}", end="")
                for j in range(self.num_labels):
                    print(f"{cm[i, j]:8d}", end="")
                print()
            
            print("-"*50 + "\n")
        
        return metrics
    
    def log_metrics(self, metrics: dict, step: int):
        """Log metrics to tensorboard."""
        self.writer.add_scalar('val/loss', metrics['loss'], step)
        self.writer.add_scalar('val/accuracy', metrics['accuracy'], step)
        self.writer.add_scalar('val/precision', metrics['precision'], step)
        self.writer.add_scalar('val/recall', metrics['recall'], step)
        self.writer.add_scalar('val/f1', metrics['f1'], step)
        
        # Log per-class metrics
        for i in range(self.num_labels):
            label_name = self.label_names[i] if i < len(self.label_names) else f"Class_{i}"
            self.writer.add_scalar(
                f'val/precision_{label_name}',
                metrics['per_class_metrics']['precision'][i],
                step
            )
            self.writer.add_scalar(
                f'val/recall_{label_name}',
                metrics['per_class_metrics']['recall'][i],
                step
            )
            self.writer.add_scalar(
                f'val/f1_{label_name}',
                metrics['per_class_metrics']['f1'][i],
                step
            )
        
        # Plot confusion matrix
        self.plot_confusion_matrix(metrics['confusion_matrix'], step)
    
    def plot_confusion_matrix(self, cm: np.ndarray, step: int):
        """Plot and save confusion matrix."""
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.label_names[:self.num_labels],
            yticklabels=self.label_names[:self.num_labels],
            ax=ax
        )
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        
        self.writer.add_figure('val/confusion_matrix', fig, step)
        
        # Save to file
        fig.savefig(os.path.join(self.output_dir, f'confusion_matrix_step{step}.png'))
        plt.close(fig)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint and training state."""
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
        
        if self.is_main_process:
            print(f"Loading checkpoint from {checkpoint_path}...")
        
        # Get the underlying model (unwrap DDP if necessary)
        model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
        
        # Load model weights
        parakeet_path = os.path.join(checkpoint_path, 'parakeet_modules.pt')
        classifier_path = os.path.join(checkpoint_path, 'classifier.pt')
        
        if os.path.exists(parakeet_path):
            parakeet_state = torch.load(parakeet_path, map_location=self.device)
            model_to_load.preprocessor.load_state_dict(parakeet_state['preprocessor'])
            model_to_load.encoder.load_state_dict(parakeet_state['encoder'])
            if self.is_main_process:
                print("  ✓ Loaded Parakeet preprocessor and encoder weights")
        
        if os.path.exists(classifier_path):
            model_to_load.classifier.load_state_dict(torch.load(classifier_path, map_location=self.device))
            if self.is_main_process:
                print("  ✓ Loaded classifier weights")
        
        # Load training state
        training_state_path = os.path.join(checkpoint_path, 'training_state.pt')
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, map_location=self.device)
            
            self.global_step = training_state.get('global_step', 0)
            self.best_val_accuracy = training_state.get('best_val_accuracy', 0.0)
            
            if 'optimizer_state_dict' in training_state:
                self.optimizer.load_state_dict(training_state['optimizer_state_dict'])
                if self.is_main_process:
                    print("  ✓ Loaded optimizer state")
            
            if 'scheduler_state_dict' in training_state:
                self.scheduler.load_state_dict(training_state['scheduler_state_dict'])
                if self.is_main_process:
                    print("  ✓ Loaded scheduler state")
            
            if self.is_main_process:
                print(f"  ✓ Resuming from step {self.global_step}")
                print(f"  ✓ Best validation accuracy: {self.best_val_accuracy:.4f}")
        else:
            if self.is_main_process:
                print("  Warning: training_state.pt not found, starting from step 0")
    
    def save_checkpoint(self, checkpoint_name: str):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.output_dir, checkpoint_name)
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Get the underlying model (unwrap DDP if necessary)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        # # Save Parakeet modules (preprocessor and encoder)
        # torch.save({
        #     'preprocessor': model_to_save.preprocessor.state_dict(),
        #     'encoder': model_to_save.encoder.state_dict(),
        # }, os.path.join(checkpoint_path, 'parakeet_modules.pt'))
        
        # Save classifier separately
        torch.save(model_to_save.classifier.state_dict(), os.path.join(checkpoint_path, 'classifier.pt'))
        if hasattr(model_to_save, 'cnn') and model_to_save.cnn is not None:
            torch.save(model_to_save.cnn.state_dict(), os.path.join(checkpoint_path, 'cnn.pt'))
            
        if hasattr(model_to_save, 'conv_layers') and model_to_save.conv_layers is not None:
            torch.save(model_to_save.conv_layers.state_dict(), os.path.join(checkpoint_path, 'conv_layers.pt'))
        
        if hasattr(model_to_save, 'transformer_encoder') and model_to_save.transformer_encoder is not None:
            torch.save(model_to_save.transformer_encoder.state_dict(), os.path.join(checkpoint_path, 'transformer_encoder.pt'))
        
        # Save training state
        torch.save({
            'global_step': self.global_step,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_accuracy': self.best_val_accuracy
        }, os.path.join(checkpoint_path, 'training_state.pt'))
        
        # Save config
        config = {
            'num_labels': self.num_labels,
            'label_names': self.label_names,
            'hidden_size': model_to_save.hidden_size,
        }
        with open(os.path.join(checkpoint_path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        if self.is_main_process:
            print(f"  ✓ Saved checkpoint to {checkpoint_path}")


def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        rank = -1
        world_size = 1
        local_rank = -1
    
    if rank != -1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        dist.barrier()
    
    return rank, local_rank, world_size


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Train Parakeet-based distortion detection model")
    
    # Data arguments
    parser.add_argument('--train_data_dir', type=str, required=True,
                       help='Directory containing training parquet files')
    parser.add_argument('--val_data_dir', type=str, required=True,
                       help='Directory containing validation parquet files')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='nvidia/parakeet-tdt-0.6b-v2',
                       help='Pre-trained Parakeet model name')
    parser.add_argument('--num_labels', type=int, default=2,
                       help='Number of distortion classes')
    parser.add_argument('--remove_deletions', action='store_true', default=False,
                       help='Remove deletions from the dataset (default: False)')
    parser.add_argument('--remove_sub', action='store_true', default=False,
                       help='Remove substitutions from the dataset (default: False)')
    parser.add_argument('--remove_ins', action='store_true', default=False,
                       help='Remove insertions from the dataset (default: False)')
    parser.add_argument('--word_level', action='store_true', default=False,
                       help='Process word-level labels (default: False)')
    parser.add_argument('--freeze_encoder', action='store_true', default=True,
                       help='Freeze the encoder (default: True)')
    parser.add_argument('--unfreeze_encoder', action='store_true',
                       help='Unfreeze the encoder for fine-tuning')
    parser.add_argument('--freeze_preprocessor', action='store_true', default=True,
                       help='Freeze the preprocessor (default: True)')
    parser.add_argument('--projector_hidden_dim', type=int, default=None,
                       help='Hidden dimension for 2-layer projector (None for single linear layer)')
    parser.add_argument('--cnn_kernel_size', type=int, default=None,
                       help='Kernel size for CNN classifier (None to disable CNN, e.g., 5 or 7)')
    parser.add_argument('--cnn_num_layers', type=int, default=2,
                       help='Number of CNN layers (default: 2)')
    parser.add_argument('--use_encoder_projection', action='store_true',
                       help='Use the encoder linear projector from Parakeet joint network (e.g., 1024 -> 640) before classifier')
    parser.add_argument('--use_conv_transformer', action='store_true',
                       help='Use Conv-Transformer architecture (2 CNN layers + Transformer layers)')
    parser.add_argument('--conv_transformer_num_heads', type=int, default=8,
                       help='Number of attention heads for Conv-Transformer (default: 8)')
    parser.add_argument('--conv_transformer_num_layers', type=int, default=3,
                       help='Number of Transformer encoder layers for Conv-Transformer (default: 3)')
    # Training arguments
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                       help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=500,
                       help='Number of warmup steps')
    parser.add_argument('--max_grad_norm', type=float, default=10,
                       help='Maximum gradient norm for clipping')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Logging interval')
    parser.add_argument('--eval_interval', type=int, default=500,
                       help='Evaluation interval')
    parser.add_argument('--save_interval', type=int, default=500,
                       help='Checkpoint saving interval')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='Path to checkpoint directory to resume training from')
    
    # Data processing arguments
    parser.add_argument('--max_length_seconds', type=float, default=30.0,
                       help='Maximum audio length in seconds')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--label_names', type=str, nargs='+',
                       default=['Clean', 'Noisy', 'RIR', 'Interference', 'Packet Loss', 'Missing'],
                       help='Names for each label class')
    
    args = parser.parse_args()
    
    # Handle freeze_encoder logic
    freeze_encoder = not args.unfreeze_encoder
    
    # Setup distributed training
    rank, local_rank, world_size = setup_distributed()
    
    # Set device
    if local_rank != -1:
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if rank == -1 or rank == 0:
        print(f"Using device: {device}")
        if world_size > 1:
            print(f"Distributed training with {world_size} GPUs")
    
    # Create datasets
    if rank == -1 or rank == 0:
        print("Loading datasets from parquet files...")
    
    # For DDP: disable dataset shuffle, let DistributedSampler handle it
    use_dataset_shuffle = (world_size <= 1)
    
    train_dataset = HFParquetDistortionDataset(
        parquet_dir=args.train_data_dir,
        feature_extractor=None,  # Parakeet handles preprocessing internally
        max_length_seconds=args.max_length_seconds,
        sample_rate=16000,
        shuffle=use_dataset_shuffle,
        seed=42,
        streaming=False,
        remove_deletions=args.remove_deletions,
        remove_substitutions=args.remove_sub,
        remove_insertions=args.remove_ins,
        word_level=args.word_level
    )
    
    val_dataset = HFParquetDistortionDataset(
        parquet_dir=args.val_data_dir,
        feature_extractor=None,
        max_length_seconds=args.max_length_seconds,
        sample_rate=16000,
        shuffle=False,
        streaming=False,
        remove_deletions=args.remove_deletions,
        remove_substitutions=args.remove_sub,
        remove_insertions=args.remove_ins,
        word_level=args.word_level
    )
    
    # Create samplers for distributed training
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn_hf_parquet,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate_fn_hf_parquet,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    if rank == -1 or rank == 0:
        print("Creating Parakeet model...")
    
    model = load_model_for_training(
        model_path=args.model_name,
        num_labels=args.num_labels,
        freeze_encoder=freeze_encoder,
        freeze_preprocessor=args.freeze_preprocessor,
        projector_hidden_dim=args.projector_hidden_dim,
        cnn_kernel_size=args.cnn_kernel_size,
        cnn_num_layers=args.cnn_num_layers,
        from_checkpoint=False,
        use_encoder_projection=args.use_encoder_projection,
        conv_transformer=args.use_conv_transformer,
        conv_transformer_num_heads=args.conv_transformer_num_heads,
        conv_transformer_num_layers=args.conv_transformer_num_layers,
    )
    
    # Move model to device and wrap with DDP if needed
    model = model.to(device)
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )
    
    # Create optimizer (only optimize trainable parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=0.01
    )
    
    total_steps = len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=total_steps,
        pct_start=min(args.warmup_steps / total_steps, 0.3),
        anneal_strategy='cos'
    )
    
    # Create trainer
    trainer = DistortionDetectionTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        num_labels=args.num_labels,
        local_rank=local_rank,
        world_size=world_size,
        label_names=args.label_names,
    )
    
    # Train
    try:
        # Resume from checkpoint if specified
        if args.resume_from_checkpoint:
            trainer.load_checkpoint(args.resume_from_checkpoint)
        trainer.train()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
