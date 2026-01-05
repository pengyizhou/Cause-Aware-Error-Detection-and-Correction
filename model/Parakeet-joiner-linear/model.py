"""
Simple linear classifier for frame-level classification.

This model takes pre-processed features [batch, time, dim=640] and applies
a simple linear classifier: ReLU -> Dropout(0.2) -> Linear(640, 2).

Uses cross-entropy loss for direct frame-level classification.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict


class ParakeetForDistortionDetection(nn.Module):
    """Simple linear classifier for frame-level classification."""
    
    def __init__(
        self,
        input_dim: int = 640,
        num_labels: int = 2,
        dropout_prob: float = 0.2,
        hidden_dim: Optional[int] = None,
        cnn_kernel_size: Optional[int] = None,
        cnn_num_layers: int = 2,
        conv_transformer: bool = False,
        conv_transformer_num_heads: int = 8,
        conv_transformer_num_layers: int = 3,
    ):
        """
        Args:
            input_dim: Input feature dimension (default: 640)
            num_labels: Number of output classes (default: 2)
            dropout_prob: Dropout probability (default: 0.2)
            hidden_dim: Optional hidden dimension for intermediate layer. If None, 
                       uses direct 640 -> 2. If provided, uses 640 -> hidden_dim -> 2.
                       Recommended values: 128, 256, or 512 for easier learning.
            cnn_kernel_size: Optional kernel size for CNN classifier. If provided,
                            uses CNN module for local window processing.
            cnn_num_layers: Number of CNN layers (default: 2). Only used if cnn_kernel_size is provided.
            conv_transformer: Whether to use Conv-Transformer architecture (2 CNN layers + Transformer layers).
            conv_transformer_num_heads: Number of attention heads for Conv-Transformer (default: 8).
            conv_transformer_num_layers: Number of Transformer encoder layers for Conv-Transformer (default: 3).
        """
        super().__init__()
        
        self.num_labels = num_labels
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        
        # Determine which architecture to use
        self.use_cnn = cnn_kernel_size is not None and not conv_transformer
        self.use_conv_transformer = conv_transformer
        
        if conv_transformer:
            # Conv-Transformer: 2 CNN layers + Transformer encoder layers + linear classifier
            conv_kernel_size = cnn_kernel_size if cnn_kernel_size is not None else 5
            conv_hidden_dim = hidden_dim if hidden_dim is not None else input_dim
            
            print(f"Using Conv-Transformer classifier: kernel_size={conv_kernel_size}, "
                  f"num_heads={conv_transformer_num_heads}, transformer_layers={conv_transformer_num_layers}")
            
            # 2 CNN layers
            cnn_layers = []
            in_channels = input_dim
            for i in range(2):  # Fixed 2 CNN layers
                out_channels = conv_hidden_dim
                cnn_layers.extend([
                    nn.Conv1d(
                        in_channels, out_channels,
                        kernel_size=conv_kernel_size,
                        padding=conv_kernel_size // 2  # Same padding to preserve sequence length
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout_prob),
                ])
                in_channels = out_channels
            self.conv_layers = nn.Sequential(*cnn_layers)
            
            # Transformer encoder layers
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=conv_hidden_dim,
                nhead=conv_transformer_num_heads,
                dim_feedforward=conv_hidden_dim * 4,
                dropout=dropout_prob,
                activation='relu',
                batch_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=conv_transformer_num_layers,
            )
            
            # Linear classifier
            self.classifier = nn.Linear(conv_hidden_dim, num_labels)
            print(f"  Conv-Transformer architecture: {input_dim} -> Conv1d x 2 -> {conv_hidden_dim} "
                  f"-> Transformer x {conv_transformer_num_layers} -> {num_labels}")
        elif cnn_kernel_size is not None:
            # CNN classifier for local window processing
            print(f"Using CNN classifier: kernel_size={cnn_kernel_size}, num_layers={cnn_num_layers}")
            cnn_hidden_dim = hidden_dim if hidden_dim is not None else input_dim
            
            cnn_layers = []
            in_channels = input_dim
            for i in range(cnn_num_layers):
                out_channels = cnn_hidden_dim
                cnn_layers.extend([
                    nn.Conv1d(
                        in_channels, out_channels, 
                        kernel_size=cnn_kernel_size, 
                        padding=cnn_kernel_size // 2  # Same padding to preserve sequence length
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout_prob),
                ])
                in_channels = out_channels
            
            self.cnn = nn.Sequential(*cnn_layers)
            self.classifier = nn.Linear(cnn_hidden_dim, num_labels)
            print(f"  CNN architecture: {input_dim} -> Conv1d x {cnn_num_layers} -> {cnn_hidden_dim} -> {num_labels}")
        elif hidden_dim is not None:
            # 2-layer: 640 -> hidden_dim -> 2
            # This is often easier to learn than direct 640 -> 2
            self.classifier = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_prob, inplace=False),
                nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_prob, inplace=False),
                nn.Linear(in_features=hidden_dim, out_features=num_labels, bias=True)
            )
            print(f"Initialized 2-layer classifier: {input_dim} -> {hidden_dim} -> {num_labels}")
        else:
            # Single layer: 640 -> 2
            self.classifier = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_prob, inplace=False),
                nn.Linear(in_features=input_dim, out_features=num_labels, bias=True)
            )
            print(f"Initialized 1-layer classifier: {input_dim} -> {num_labels}")
        
        # Initialize classifier weights
        self._init_weights()
        
        # Loss function
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
        
        print(f"Initialized classifier: input_dim={input_dim}, num_labels={num_labels}, dropout={dropout_prob}")
    
    def _init_weights(self):
        """Initialize classifier weights."""
        # Initialize linear classifier
        if isinstance(self.classifier, nn.Linear):
            self.classifier.weight.data.normal_(mean=0.0, std=0.02)
            if self.classifier.bias is not None:
                self.classifier.bias.data.zero_()
        elif isinstance(self.classifier, nn.Sequential):
            for module in self.classifier:
                if isinstance(module, nn.Linear):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                    if module.bias is not None:
                        module.bias.data.zero_()
        
        # Initialize CNN layers if present
        if self.use_cnn:
            for module in self.cnn:
                if isinstance(module, nn.Conv1d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm1d):
                    module.weight.data.fill_(1.0)
                    module.bias.data.zero_()
        
        # Initialize Conv-Transformer layers if present
        if self.use_conv_transformer:
            for module in self.conv_layers:
                if isinstance(module, nn.Conv1d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm1d):
                    module.weight.data.fill_(1.0)
                    module.bias.data.zero_()
            # Transformer layers use PyTorch default initialization
    
    def forward(
        self,
        input_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        return_labels: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_values: Input features tensor [batch_size, time, dim=640]
            labels: Frame-level labels [batch_size, time] (optional)
            return_dict: Whether to return a dictionary
            return_labels: Whether to return processed labels
            
        Returns:
            Dictionary containing:
            - loss: Cross-entropy loss (if labels provided)
            - logits: Frame-level predictions [batch_size, time, num_labels]
            - labels: Flattened labels (if return_labels=True)
        """
        # Input: [batch, time, dim=640]
        # Apply classifier based on architecture
        if self.use_conv_transformer:
            # Conv-Transformer: CNN layers -> Transformer encoder -> classifier
            # CNN expects [batch, channels, seq_len], so transpose
            conv_input = input_values.transpose(1, 2)  # [batch, input_dim, time]
            conv_output = self.conv_layers(conv_input)  # [batch, conv_hidden_dim, time]
            conv_output = conv_output.transpose(1, 2)  # [batch, time, conv_hidden_dim]
            
            # Transformer encoder (batch_first=True, so input is [batch, time, hidden_dim])
            transformer_output = self.transformer_encoder(conv_output)  # [batch, time, conv_hidden_dim]
            
            logits = self.classifier(transformer_output)  # [batch, time, num_labels]
        elif self.use_cnn:
            # CNN expects [batch, channels, seq_len], so transpose
            cnn_input = input_values.transpose(1, 2)  # [batch, input_dim, time]
            cnn_output = self.cnn(cnn_input)  # [batch, cnn_hidden_dim, time]
            cnn_output = cnn_output.transpose(1, 2)  # [batch, time, cnn_hidden_dim]
            logits = self.classifier(cnn_output)  # [batch, time, num_labels]
        else:
            # Linear classifier
            logits = self.classifier(input_values)  # [batch, time, num_labels]
        
        loss = None
        labels_flat = None
        if labels is not None:
            loss, labels_flat = self.compute_loss(logits, labels)
        
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output
        
        return {
            'loss': loss,
            'logits': logits,
            'labels': labels_flat if return_labels else None,
        }
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for framewise classification.
        
        Args:
            logits: Model predictions [batch_size, time, num_labels]
            labels: Ground truth labels [batch_size, time]
            
        Returns:
            Computed loss (scalar), flattened labels
        """
        batch_size, num_frames, num_labels = logits.shape
        
        # Align labels with logits length
        if num_frames < labels.shape[1]:
            labels = labels[:, :num_frames]
        elif num_frames > labels.shape[1]:
            # Pad labels with -100 (ignore index)
            pad_size = num_frames - labels.shape[1]
            pad_tensor = torch.full(
                (batch_size, pad_size), -100, 
                dtype=labels.dtype, device=labels.device
            )
            labels = torch.cat([labels, pad_tensor], dim=1)
        
        # Flatten logits and labels
        logits_flat = logits.reshape(-1, num_labels)
        labels_flat = labels.reshape(-1)
        
        # Compute loss
        loss = self.loss_fct(logits_flat, labels_flat)
        
        return loss, labels_flat
    
    def get_trainable_parameters(self) -> int:
        """Get the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_parameters(self) -> int:
        """Get the total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def load_model_for_training(
    input_dim: int = 640,
    num_labels: int = 2,
    dropout_prob: float = 0.2,
    hidden_dim: Optional[int] = None,
    cnn_kernel_size: Optional[int] = None,
    cnn_num_layers: int = 2,
    conv_transformer: bool = False,
    conv_transformer_num_heads: int = 8,
    conv_transformer_num_layers: int = 3,
) -> nn.Module:
    """
    Load model for training.
    
    Args:
        input_dim: Input feature dimension (default: 640)
        num_labels: Number of output classes (default: 2)
        dropout_prob: Dropout probability (default: 0.2)
        hidden_dim: Optional hidden dimension for intermediate layer. If None, 
                   uses direct input_dim -> num_labels (1 layer). If provided, 
                   uses input_dim -> hidden_dim -> num_labels (2 layers).
        cnn_kernel_size: Optional kernel size for CNN classifier. If provided,
                        uses CNN module for local window processing.
        cnn_num_layers: Number of CNN layers (default: 2). Only used if cnn_kernel_size is provided.
        conv_transformer: Whether to use Conv-Transformer architecture (2 CNN layers + Transformer layers).
        conv_transformer_num_heads: Number of attention heads for Conv-Transformer (default: 8).
        conv_transformer_num_layers: Number of Transformer encoder layers for Conv-Transformer (default: 3).
        
    Returns:
        Model ready for training
    """
    model = ParakeetForDistortionDetection(
        input_dim=input_dim,
        num_labels=num_labels,
        dropout_prob=dropout_prob,
        hidden_dim=hidden_dim,
        cnn_kernel_size=cnn_kernel_size,
        cnn_num_layers=cnn_num_layers,
        conv_transformer=conv_transformer,
        conv_transformer_num_heads=conv_transformer_num_heads,
        conv_transformer_num_layers=conv_transformer_num_layers,
    )
    
    print(f"Total parameters: {model.get_total_parameters():,}")
    print(f"Trainable parameters: {model.get_trainable_parameters():,}")
    
    return model


if __name__ == "__main__":
    # Test model
    print("Testing linear classifier model...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Test model
    print("\n" + "="*50)
    print("Testing linear classifier architecture...")
    print("="*50)
    model = ParakeetForDistortionDetection(
        input_dim=640,
        num_labels=2,
        dropout_prob=0.2,
    ).to(device)
    
    # Create dummy input: [batch, time, dim=640]
    batch_size = 2
    time_steps = 100
    input_dim = 640
    input_values = torch.randn(batch_size, time_steps, input_dim).to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_values)
    
    print(f"\nInput shape: {input_values.shape}")
    print(f"Output logits shape: {outputs['logits'].shape}")
    print(f"Number of time steps: {outputs['logits'].shape[1]}")
    print(f"Number of classes: {outputs['logits'].shape[2]}")
    print(f"\nModel ready for training!")
