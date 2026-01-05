"""
Parakeet-Encoder model with linear projector for frame-level classification.

This model uses NVIDIA's Parakeet TDT encoder with a linear classification head
to predict frame-level distortion types (0-5):
- 0: Clean speech
- 1: Noisy background
- 2: Room Impulse Response (RIR)
- 3: Interference speakers
- 4: Network packet loss / low bitrate codec
- 5: Totally missing segments

Uses cross-entropy loss for direct frame-level classification.
"""

import torch
import torch.nn as nn
import nemo.collections.asr as nemo_asr
from typing import Optional, Dict
import ipdb


class ParakeetForDistortionDetection(nn.Module):
    """Parakeet encoder model with linear projector for distortion detection."""
    
    def __init__(
        self,
        parakeet_model_name: str = "nvidia/parakeet-tdt-0.6b-v2",
        num_labels: int = 6,
        freeze_encoder: bool = True,
        freeze_preprocessor: bool = True,
        hidden_dropout_prob: float = 0.1,
        projector_hidden_dim: Optional[int] = None,
        cnn_kernel_size: Optional[int] = None,
        cnn_num_layers: int = 2,
        use_encoder_projection: bool = False,
        conv_transformer: bool = False,
        conv_transformer_num_heads: int = 8,
        conv_transformer_num_layers: int = 3,
    ):
        """
        Args:
            parakeet_model_name: Pre-trained Parakeet model name
            num_labels: Number of distortion classes (default: 6)
            freeze_encoder: Whether to freeze the encoder
            freeze_preprocessor: Whether to freeze the preprocessor
            hidden_dropout_prob: Dropout probability for the classification head
            projector_hidden_dim: Optional hidden dimension for 2-layer projector.
                                  If None, uses single linear layer.
            cnn_kernel_size: Optional kernel size for CNN classifier. If provided,
                             uses CNN module for local window processing.
            cnn_num_layers: Number of CNN layers (default: 2). Only used if cnn_kernel_size is provided.
            use_encoder_projection: Whether to use the encoder linear projector from Parakeet's
                                    joint network (e.g., 1024 -> 640) before the classifier.
            conv_transformer: Whether to use Conv-Transformer architecture (2 CNN layers + 3 Transformer layers).
            conv_transformer_num_heads: Number of attention heads for Conv-Transformer (default: 8).
            conv_transformer_num_layers: Number of Transformer encoder layers for Conv-Transformer (default: 3).
        """
        super().__init__()
        
        self.num_labels = num_labels
        self.parakeet_model_name = parakeet_model_name
        
        # Load pre-trained Parakeet model
        print(f"Loading Parakeet model from {parakeet_model_name}...")
        asr_model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=parakeet_model_name
        )
        
        # Extract preprocessor and encoder
        self.preprocessor = asr_model.preprocessor
        self.encoder = asr_model.encoder
        
        # Get encoder hidden size (Parakeet TDT 0.6B has 1024 hidden dim)
        encoder_hidden_size = self.encoder.d_model if hasattr(self.encoder, 'd_model') else 1024
        print(f"Encoder hidden size: {encoder_hidden_size}")
        
        # Optionally extract encoder linear projector from joint network
        self.use_encoder_projection = use_encoder_projection
        self.encoder_projection = None
        
        if use_encoder_projection:
            # Try to access the joint network's encoder linear projector
            # Common paths: asr_model.decoder.joint.enc or asr_model.joint.enc
            encoder_proj = None
            if hasattr(asr_model, 'decoder') and hasattr(asr_model.decoder, 'joint'):
                if hasattr(asr_model.decoder.joint, 'enc'):
                    encoder_proj = asr_model.decoder.joint.enc
                    print(f"Found encoder projection in decoder.joint.enc")
            elif hasattr(asr_model, 'joint') and hasattr(asr_model.joint, 'enc'):
                encoder_proj = asr_model.joint.enc
                print(f"Found encoder projection in joint.enc")
            
            if encoder_proj is not None:
                # Extract the linear layer
                self.encoder_projection = encoder_proj
                # Get output dimension from the projection
                if hasattr(encoder_proj, 'out_features'):
                    self.hidden_size = encoder_proj.out_features
                elif hasattr(encoder_proj, 'weight'):
                    self.hidden_size = encoder_proj.weight.shape[0]
                else:
                    self.hidden_size = encoder_hidden_size
                    print(f"Warning: Could not determine projection output size, using encoder size")
                
                print(f"Using Parakeet encoder projection: {encoder_hidden_size} -> {self.hidden_size}")
                print("Freezing encoder projection...")
                self.freeze_encoder_projection()
            else:
                print(f"Warning: use_encoder_projection=True but encoder projection not found in model. Using encoder size directly.")
                self.hidden_size = encoder_hidden_size
                self.use_encoder_projection = False
        else:
            self.hidden_size = encoder_hidden_size
        
        print("Freezing preprocessor...")
        self.freeze_preprocessor()
        
        print("Freezing encoder...")
        self.freeze_encoder()
        
        # Linear projector/classifier head
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.use_cnn = cnn_kernel_size is not None and not conv_transformer
        self.use_conv_transformer = conv_transformer
        
        if conv_transformer:
            # Conv-Transformer: 2 CNN layers + Transformer encoder layers + linear classifier
            conv_kernel_size = cnn_kernel_size if cnn_kernel_size is not None else 5
            conv_hidden_dim = projector_hidden_dim if projector_hidden_dim is not None else self.hidden_size
            
            print(f"Using Conv-Transformer classifier: kernel_size={conv_kernel_size}, "
                  f"num_heads={conv_transformer_num_heads}, transformer_layers={conv_transformer_num_layers}")
            
            # 2 CNN layers
            cnn_layers = []
            in_channels = self.hidden_size
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
                    nn.Dropout(hidden_dropout_prob),
                ])
                in_channels = out_channels
            self.conv_layers = nn.Sequential(*cnn_layers)
            
            # Transformer encoder layers
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=conv_hidden_dim,
                nhead=conv_transformer_num_heads,
                dim_feedforward=conv_hidden_dim * 4,
                dropout=hidden_dropout_prob,
                activation='relu',
                batch_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=conv_transformer_num_layers,
            )
            
            # Linear classifier
            self.classifier = nn.Linear(conv_hidden_dim, num_labels)
            print(f"  Conv-Transformer architecture: {self.hidden_size} -> Conv1d x 2 -> {conv_hidden_dim} "
                  f"-> Transformer x {conv_transformer_num_layers} -> {num_labels}")
        elif cnn_kernel_size is not None:
            # CNN classifier for local window processing
            # Input: [batch, seq_len, hidden_size] -> process with 1D CNN
            print(f"Using CNN classifier: kernel_size={cnn_kernel_size}, num_layers={cnn_num_layers}")
            cnn_hidden_dim = projector_hidden_dim if projector_hidden_dim is not None else self.hidden_size
            
            cnn_layers = []
            in_channels = self.hidden_size
            for i in range(cnn_num_layers):
                out_channels = cnn_hidden_dim if i < cnn_num_layers - 1 else cnn_hidden_dim
                cnn_layers.extend([
                    nn.Conv1d(
                        in_channels, out_channels, 
                        kernel_size=cnn_kernel_size, 
                        padding=cnn_kernel_size // 2  # Same padding to preserve sequence length
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(hidden_dropout_prob),
                ])
                in_channels = out_channels
            
            self.cnn = nn.Sequential(*cnn_layers)
            self.classifier = nn.Linear(cnn_hidden_dim, num_labels)
            print(f"  CNN architecture: {self.hidden_size} -> Conv1d x {cnn_num_layers} -> {cnn_hidden_dim} -> {num_labels}")
        elif projector_hidden_dim is not None:
            # 2-layer projector: hidden_size -> projector_hidden_dim -> num_labels
            print(f"Using 2-layer projector: {self.hidden_size} -> {projector_hidden_dim} -> {num_labels}")
            self.classifier = nn.Sequential(
                nn.Linear(self.hidden_size, projector_hidden_dim),
                nn.ReLU(),
                nn.Dropout(hidden_dropout_prob),
                nn.Linear(projector_hidden_dim, num_labels)
            )
        else:
            # Single linear layer: hidden_size -> num_labels
            print(f"Using single linear layer: {self.hidden_size} -> {num_labels}")
            self.classifier = nn.Linear(self.hidden_size, num_labels)
        
        # Initialize classifier weights
        self._init_weights()
        
        # Loss function
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
        
        # Delete the original model to free memory
        del asr_model
        torch.cuda.empty_cache()
    
    def _init_weights(self):
        """Initialize classifier weights."""
        # Initialize classifier
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
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        return_labels: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_values: Audio input tensor [batch_size, sequence_length]
            attention_mask: Attention mask [batch_size, sequence_length]
            labels: Frame-level labels [batch_size, num_frames]
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return a dictionary
            return_labels: Whether to return processed labels
            
        Returns:
            Dictionary containing:
            - loss: Cross-entropy loss (if labels provided)
            - logits: Frame-level predictions [batch_size, num_frames, num_labels]
            - hidden_states: Hidden states (if requested)
            - labels: Flattened labels (if return_labels=True)
        """
        batch_size = input_values.shape[0]
        
        # Compute audio lengths from attention mask
        if attention_mask is not None:
            audio_lengths = attention_mask.sum(dim=-1).long()
        else:
            audio_lengths = torch.full(
                (batch_size,), 
                input_values.shape[1], 
                dtype=torch.long, 
                device=input_values.device
            )
        
        # Step 1: Audio -> Mel spectrogram
        mel_features, mel_lengths = self.preprocessor(
            input_signal=input_values,
            length=audio_lengths
        )
        # mel_features: [batch, mel_dim, time]
        
        # Step 2: Mel -> Encoder embeddings
        encoder_output, encoder_lengths = self.encoder(
            audio_signal=mel_features,
            length=mel_lengths
        )
        # encoder_output: [batch, hidden_size, time/subsampling_factor] from NeMo encoder
        # Need to transpose to [batch, time/subsampling_factor, hidden_size] for classifier
        
        # Transpose from [batch, hidden_size, seq_len] to [batch, seq_len, hidden_size]
        hidden_states = encoder_output.transpose(1, 2)
        
        # Apply Parakeet encoder projection if enabled
        if self.use_encoder_projection and self.encoder_projection is not None:
            hidden_states = self.encoder_projection(hidden_states)
        
        # Apply dropout and classifier
        hidden_states_dropped = self.dropout(hidden_states)
        
        if self.use_conv_transformer:
            # Conv-Transformer: CNN layers -> Transformer encoder -> classifier
            # CNN expects [batch, channels, seq_len], so transpose
            conv_input = hidden_states_dropped.transpose(1, 2)  # [batch, hidden_size, seq_len]
            conv_output = self.conv_layers(conv_input)  # [batch, conv_hidden_dim, seq_len]
            conv_output = conv_output.transpose(1, 2)  # [batch, seq_len, conv_hidden_dim]
            
            # Transformer encoder (batch_first=True, so input is [batch, seq_len, hidden_dim])
            transformer_output = self.transformer_encoder(conv_output)  # [batch, seq_len, conv_hidden_dim]
            
            logits = self.classifier(transformer_output)  # [batch, num_frames, num_labels]
        elif self.use_cnn:
            # CNN expects [batch, channels, seq_len], so transpose
            cnn_input = hidden_states_dropped.transpose(1, 2)  # [batch, hidden_size, seq_len]
            cnn_output = self.cnn(cnn_input)  # [batch, cnn_hidden_dim, seq_len]
            cnn_output = cnn_output.transpose(1, 2)  # [batch, seq_len, cnn_hidden_dim]
            logits = self.classifier(cnn_output)  # [batch, num_frames, num_labels]
        else:
            logits = self.classifier(hidden_states_dropped)  # [batch, num_frames, num_labels]
        
        loss = None
        labels_flat = None
        if labels is not None:
            loss, labels_flat = self.compute_loss(logits, labels)
        
        if not return_dict:
            output = (logits,)
            if output_hidden_states:
                output += (encoder_output,)
            return ((loss,) + output) if loss is not None else output
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': encoder_output if output_hidden_states else None,
            'labels': labels_flat if return_labels else None,
            'encoder_lengths': encoder_lengths
        }
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for framewise classification.
        
        Args:
            logits: Model predictions [batch_size, num_frames, num_labels]
            labels: Ground truth labels [batch_size, num_label_frames]
            
        Returns:
            Computed loss (scalar), flattened labels
        """
        batch_size, num_frames, num_labels = logits.shape
        
        # assert num_frames == labels.shape[1], f"Number of frames must match number of label frames, got {num_frames} and {labels.shape[1]}"
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
    
    def freeze_encoder_projection(self):
        """Freeze the encoder projection."""
        if self.encoder_projection is not None:
            for param in self.encoder_projection.parameters():
                param.requires_grad = False
    
    def freeze_encoder(self):
        """Freeze the encoder."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze the encoder."""
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def freeze_preprocessor(self):
        """Freeze the preprocessor."""
        for param in self.preprocessor.parameters():
            param.requires_grad = False
    
    def unfreeze_preprocessor(self):
        """Unfreeze the preprocessor."""
        for param in self.preprocessor.parameters():
            param.requires_grad = True
    
    def get_trainable_parameters(self) -> int:
        """Get the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_parameters(self) -> int:
        """Get the total number of parameters."""
        return sum(p.numel() for p in self.parameters())


class ParakeetForDistortionDetectionFromCheckpoint(ParakeetForDistortionDetection):
    """
    Load Parakeet from a saved checkpoint instead of from HuggingFace.
    Useful when you have a fine-tuned model or want to load from local files.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        num_labels: int = 6,
        freeze_encoder: bool = True,
        freeze_preprocessor: bool = True,
        hidden_dropout_prob: float = 0.1,
        projector_hidden_dim: Optional[int] = None,
        cnn_kernel_size: Optional[int] = None,
        cnn_num_layers: int = 2,
        use_encoder_projection: bool = False,
        conv_transformer: bool = False,
        conv_transformer_num_heads: int = 8,
        conv_transformer_num_layers: int = 3,
    ):
        """
        Args:
            checkpoint_path: Path to saved checkpoint (.pt file with preprocessor and encoder state dicts)
            num_labels: Number of distortion classes
            freeze_encoder: Whether to freeze the encoder
            freeze_preprocessor: Whether to freeze the preprocessor
            hidden_dropout_prob: Dropout probability
            projector_hidden_dim: Optional hidden dimension for 2-layer projector
            cnn_kernel_size: Optional kernel size for CNN classifier
            cnn_num_layers: Number of CNN layers (default: 2)
            use_encoder_projection: Whether to use the encoder linear projector from Parakeet's
                                    joint network (e.g., 1024 -> 640) before the classifier.
            conv_transformer: Whether to use Conv-Transformer architecture (2 CNN layers + 3 Transformer layers).
            conv_transformer_num_heads: Number of attention heads for Conv-Transformer (default: 8).
            conv_transformer_num_layers: Number of Transformer encoder layers for Conv-Transformer (default: 3).
        """
        # Skip parent __init__ and initialize nn.Module directly
        nn.Module.__init__(self)
        
        self.num_labels = num_labels
        
        print(f"Loading Parakeet model architecture...")
        # First load the architecture from NeMo
        asr_model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/parakeet-tdt-0.6b-v2"
        )
        
        # Load saved state_dicts if provided
        if checkpoint_path:
            print(f"Loading checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            asr_model.preprocessor.load_state_dict(checkpoint['preprocessor'])
            asr_model.encoder.load_state_dict(checkpoint['encoder'])
        
        # Extract preprocessor and encoder
        self.preprocessor = asr_model.preprocessor
        self.encoder = asr_model.encoder
        
        # Get encoder hidden size
        encoder_hidden_size = self.encoder.d_model if hasattr(self.encoder, 'd_model') else 1024
        print(f"Encoder hidden size: {encoder_hidden_size}")
        
        # Optionally extract encoder linear projector from joint network
        self.use_encoder_projection = use_encoder_projection
        self.encoder_projection = None
        
        if use_encoder_projection:
            # Try to access the joint network's encoder linear projector
            encoder_proj = None
            if hasattr(asr_model, 'decoder') and hasattr(asr_model.decoder, 'joint'):
                if hasattr(asr_model.decoder.joint, 'enc'):
                    encoder_proj = asr_model.decoder.joint.enc
                    print(f"Found encoder projection in decoder.joint.enc")
            elif hasattr(asr_model, 'joint') and hasattr(asr_model.joint, 'enc'):
                encoder_proj = asr_model.joint.enc
                print(f"Found encoder projection in joint.enc")
            
            if encoder_proj is not None:
                # Extract the linear layer
                self.encoder_projection = encoder_proj
                # Get output dimension from the projection
                if hasattr(encoder_proj, 'out_features'):
                    self.hidden_size = encoder_proj.out_features
                elif hasattr(encoder_proj, 'weight'):
                    self.hidden_size = encoder_proj.weight.shape[0]
                else:
                    self.hidden_size = encoder_hidden_size
                    print(f"Warning: Could not determine projection output size, using encoder size")
                
                print(f"Using Parakeet encoder projection: {encoder_hidden_size} -> {self.hidden_size}")
                # Freeze the encoder projection
                for param in self.encoder_projection.parameters():
                    param.requires_grad = False
            else:
                print(f"Warning: use_encoder_projection=True but encoder projection not found in model. Using encoder size directly.")
                self.hidden_size = encoder_hidden_size
                self.use_encoder_projection = False
        else:
            self.hidden_size = encoder_hidden_size
        
        # Freeze preprocessor if specified
        if freeze_preprocessor:
            print("Freezing preprocessor...")
            for param in self.preprocessor.parameters():
                param.requires_grad = False
        
        # Freeze encoder if specified
        if freeze_encoder:
            print("Freezing encoder...")
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Linear projector/classifier head
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.use_cnn = cnn_kernel_size is not None and not conv_transformer
        self.use_conv_transformer = conv_transformer
        
        if conv_transformer:
            # Conv-Transformer: 2 CNN layers + Transformer encoder layers + linear classifier
            conv_kernel_size = cnn_kernel_size if cnn_kernel_size is not None else 5
            conv_hidden_dim = projector_hidden_dim if projector_hidden_dim is not None else self.hidden_size
            
            print(f"Using Conv-Transformer classifier: kernel_size={conv_kernel_size}, "
                  f"num_heads={conv_transformer_num_heads}, transformer_layers={conv_transformer_num_layers}")
            
            # 2 CNN layers
            cnn_layers = []
            in_channels = self.hidden_size
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
                    nn.Dropout(hidden_dropout_prob),
                ])
                in_channels = out_channels
            self.conv_layers = nn.Sequential(*cnn_layers)
            
            # Transformer encoder layers
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=conv_hidden_dim,
                nhead=conv_transformer_num_heads,
                dim_feedforward=conv_hidden_dim * 4,
                dropout=hidden_dropout_prob,
                activation='relu',
                batch_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=conv_transformer_num_layers,
            )
            
            # Linear classifier
            self.classifier = nn.Linear(conv_hidden_dim, num_labels)
            print(f"  Conv-Transformer architecture: {self.hidden_size} -> Conv1d x 2 -> {conv_hidden_dim} "
                  f"-> Transformer x {conv_transformer_num_layers} -> {num_labels}")
        elif cnn_kernel_size is not None:
            # CNN classifier for local window processing
            print(f"Using CNN classifier: kernel_size={cnn_kernel_size}, num_layers={cnn_num_layers}")
            cnn_hidden_dim = projector_hidden_dim if projector_hidden_dim is not None else self.hidden_size
            
            cnn_layers = []
            in_channels = self.hidden_size
            for i in range(cnn_num_layers):
                out_channels = cnn_hidden_dim if i < cnn_num_layers - 1 else cnn_hidden_dim
                cnn_layers.extend([
                    nn.Conv1d(
                        in_channels, out_channels, 
                        kernel_size=cnn_kernel_size, 
                        padding=cnn_kernel_size // 2
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(hidden_dropout_prob),
                ])
                in_channels = out_channels
            
            self.cnn = nn.Sequential(*cnn_layers)
            self.classifier = nn.Linear(cnn_hidden_dim, num_labels)
            print(f"  CNN architecture: {self.hidden_size} -> Conv1d x {cnn_num_layers} -> {cnn_hidden_dim} -> {num_labels}")
        elif projector_hidden_dim is not None:
            print(f"Using 2-layer projector: {self.hidden_size} -> {projector_hidden_dim} -> {num_labels}")
            self.classifier = nn.Sequential(
                nn.Linear(self.hidden_size, projector_hidden_dim),
                nn.ReLU(),
                nn.Dropout(hidden_dropout_prob),
                nn.Linear(projector_hidden_dim, num_labels)
            )
        else:
            print(f"Using single linear layer: {self.hidden_size} -> {num_labels}")
            self.classifier = nn.Linear(self.hidden_size, num_labels)
        
        # Initialize classifier weights
        self._init_weights()
        
        # Loss function
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
        
        # Clean up
        del asr_model
        torch.cuda.empty_cache()


def load_model_for_training(
    model_path: str = "nvidia/parakeet-tdt-0.6b-v2",
    num_labels: int = 6,
    freeze_encoder: bool = True,
    freeze_preprocessor: bool = True,
    projector_hidden_dim: Optional[int] = None,
    cnn_kernel_size: Optional[int] = None,
    cnn_num_layers: int = 2,
    from_checkpoint: bool = False,
    use_encoder_projection: bool = False,
    conv_transformer: bool = False,
    conv_transformer_num_heads: int = 8,
    conv_transformer_num_layers: int = 3,
) -> nn.Module:
    """
    Load model for training.
    
    Args:
        model_path: Path to pre-trained Parakeet model or checkpoint
        num_labels: Number of distortion classes
        freeze_encoder: Whether to freeze encoder
        freeze_preprocessor: Whether to freeze preprocessor
        projector_hidden_dim: Optional hidden dim for 2-layer projector
        cnn_kernel_size: Optional kernel size for CNN classifier
        cnn_num_layers: Number of CNN layers (default: 2)
        from_checkpoint: Whether to load from saved checkpoint
        use_encoder_projection: Whether to use the encoder linear projector from Parakeet's
                                joint network (e.g., 1024 -> 640) before the classifier.
        conv_transformer: Whether to use Conv-Transformer architecture (2 CNN layers + Transformer layers).
        conv_transformer_num_heads: Number of attention heads for Conv-Transformer (default: 8).
        conv_transformer_num_layers: Number of Transformer encoder layers for Conv-Transformer (default: 3).
        
    Returns:
        Model ready for training
    """
    if from_checkpoint:
        model = ParakeetForDistortionDetectionFromCheckpoint(
            checkpoint_path=model_path,
            num_labels=num_labels,
            freeze_encoder=freeze_encoder,
            freeze_preprocessor=freeze_preprocessor,
            projector_hidden_dim=projector_hidden_dim,
            cnn_kernel_size=cnn_kernel_size,
            cnn_num_layers=cnn_num_layers,
            use_encoder_projection=use_encoder_projection,
            conv_transformer=conv_transformer,
            conv_transformer_num_heads=conv_transformer_num_heads,
            conv_transformer_num_layers=conv_transformer_num_layers,
        )
    else:
        model = ParakeetForDistortionDetection(
            parakeet_model_name=model_path,
            num_labels=num_labels,
            freeze_encoder=freeze_encoder,
            freeze_preprocessor=freeze_preprocessor,
            projector_hidden_dim=projector_hidden_dim,
            cnn_kernel_size=cnn_kernel_size,
            cnn_num_layers=cnn_num_layers,
            use_encoder_projection=use_encoder_projection,
            conv_transformer=conv_transformer,
            conv_transformer_num_heads=conv_transformer_num_heads,
            conv_transformer_num_layers=conv_transformer_num_layers,
        )
    
    print(f"Total parameters: {model.get_total_parameters():,}")
    print(f"Trainable parameters: {model.get_trainable_parameters():,}")
    
    return model


if __name__ == "__main__":
    # Test model
    print("Testing Parakeet model architecture...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Test CNN architecture
    print("\n" + "="*50)
    print("Testing CNN classifier architecture...")
    print("="*50)
    model = ParakeetForDistortionDetection(
        parakeet_model_name="nvidia/parakeet-tdt-0.6b-v2",
        num_labels=6,
        freeze_encoder=True,
        freeze_preprocessor=True,
        projector_hidden_dim=512,
        cnn_kernel_size=5,  # CNN with kernel size 5
        cnn_num_layers=2,
    ).to(device)
    
    # Create dummy input
    batch_size = 2
    sequence_length = 16000 * 3  # 3 seconds at 16kHz
    input_values = torch.randn(batch_size, sequence_length).to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_values)
    
    print(f"\nInput shape: {input_values.shape}")
    print(f"Output logits shape: {outputs['logits'].shape}")
    print(f"Number of frames: {outputs['logits'].shape[1]}")
    print(f"Encoder lengths: {outputs['encoder_lengths']}")
    print(f"\nModel ready for training!")
