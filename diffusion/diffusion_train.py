import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter
import os
from datetime import datetime
from dotenv import load_dotenv
from src import GetOneHot, GetFasta 
from src.Block import Block, get_activation
from src.Layer import Layer


class CategoricalTransition:
    """Categorical diffusion transition matrices for amino acid sequences"""
    
    def __init__(self, num_steps, num_classes=21):
        self.num_steps = num_steps
        self.num_classes = num_classes
        
        # Beta schedule controls how much noise is added
        betas = torch.linspace(0.0001, 0.02, num_steps)
        
        # Q_t[i,j] = probability of transitioning from class i to class j
        self.Q_t = []
        self.Q_bar_t = []  #Cumulative
        
        for t in range(num_steps):
            beta_t = betas[t]
            # Transition matrix at time t
            Q = torch.eye(num_classes) * (1 - beta_t) + torch.ones(num_classes, num_classes) * (beta_t / num_classes)
            self.Q_t.append(Q)
            
            # Cumulative product Q_bar_t = Q_1 * Q_2 * ... * Q_t
            if t == 0:
                self.Q_bar_t.append(Q)
            else:
                self.Q_bar_t.append(self.Q_bar_t[-1] @ Q)
    
    def add_noise(self, x_0, t, device):
        """
        Add noise to sequences x_0 at timestep t
        Vectorized implementation
        """
        batch_size, seq_len = x_0.shape
        x_t = torch.empty_like(x_0)

        for idx in range(batch_size):
            Q_bar = self.Q_bar_t[t[idx]].to(device)  # (num_classes, num_classes)
            seq = x_0[idx]  # (seq_len,)
            probs = Q_bar[seq]  # (seq_len, num_classes)

            # Sample all positions at once for this sequence
            samples = torch.multinomial(probs, num_samples=1).squeeze(1)  
            x_t[idx] = samples

        return x_t

    
    def posterior(self, x_t, x_0, t, device):
        """
        Fully vectorized posterior q(x_{t-1} | x_t, x_0)
        Inputs:
            x_t: (batch, seq_len)
            x_0: (batch, seq_len)
            t:   (batch,)
        Output:
            posterior: (batch, seq_len, num_classes)
        """
        batch_size, seq_len = x_t.shape
        num_classes = self.num_classes

        posterior = torch.zeros(batch_size, seq_len, num_classes, device=device)

        # Create a mask for t == 0 and t > 0
        t_0_mask = t == 0
        t_pos_mask = ~t_0_mask

        # Case 1: t == 0 → deterministic one-hot
        if t_0_mask.any():
            idxs = torch.nonzero(t_0_mask).squeeze(1)
            posterior[idxs] = F.one_hot(x_0[idxs], num_classes=num_classes).float()

        # Case 2: t > 0
        if t_pos_mask.any():
            idxs = torch.nonzero(t_pos_mask).squeeze(1)

            x_0_pos = x_0[idxs]  # (n, seq_len)
            x_t_pos = x_t[idxs]
            t_pos = t[idxs]

            # Preallocate batch tensor
            posterior_pos = torch.zeros(len(idxs), seq_len, num_classes, device=device)

            # Do per-timestep block with batching inside
            for unique_t in torch.unique(t_pos):
                sel = (t_pos == unique_t)
                batch_indices = torch.nonzero(sel).squeeze(1)

                Q = self.Q_t[unique_t.item()].to(device)               # (C, C)
                Q_bar_prev = self.Q_bar_t[unique_t.item() - 1].to(device)  # (C, C)

                x_0_sub = x_0_pos[sel]  # (b', seq_len)
                x_t_sub = x_t_pos[sel]  # (b', seq_len)

                # q_prev: (b', seq_len, C)
                q_prev = Q_bar_prev[x_0_sub]

                # q_forward: (b', seq_len, C)
                q_forward = Q[:, x_t_sub.T].permute(2, 1, 0)  # [C, seq_len, b'] → [b', seq_len, C]

                # Multiply and normalize
                post = q_prev * q_forward
                post = post / post.sum(dim=-1, keepdim=True)

                posterior_pos[sel] = post

            # Place into main posterior tensor
            posterior[idxs] = posterior_pos

        return posterior

    
    def denoise(self, x_t, pred_probs, t, temperature=1.0):
        """
        Sample x_{t-1} given x_t and predicted probabilities
        pred_probs: (batch, seq_len, num_classes) - predicted probabilities
        """
        batch_size, seq_len = x_t.shape
        x_prev = torch.zeros_like(x_t)
        
        # Apply temperature
        if temperature != 1.0:
            pred_probs = pred_probs.pow(1.0 / temperature)
            pred_probs = pred_probs / pred_probs.sum(dim=-1, keepdim=True)
        
        for b in range(batch_size):
            for i in range(seq_len):
                probs = pred_probs[b, i]
                x_prev[b, i] = torch.multinomial(probs, 1)
        
        return x_prev


class ImprovedTimeEmbedding(nn.Module):
    """Enhanced time embedding with multiple features"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear = nn.Sequential(
            nn.Linear(dim + 2, dim * 2),  # +2 for sin/cos features
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )
    
    def forward(self, t, max_t):
        # Sinusoidal embedding
        half = self.dim // 2
        emb = torch.exp(torch.arange(half, device=t.device) * -math.log(10000) / (half-1))
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        
        # Add normalized time features
        t_norm = t.float() / max_t
        t_features = torch.stack([
            torch.sin(2 * math.pi * t_norm),
            torch.cos(2 * math.pi * t_norm)
        ], dim=1)
        
        # Combine features
        emb = torch.cat([emb, t_features], dim=1)
        return self.linear(emb)


class ConditionalLayerNorm(nn.Module):
    """Layer normalization conditioned on time and position type"""
    def __init__(self, embed_dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.scale_proj = nn.Linear(cond_dim, embed_dim)
        self.shift_proj = nn.Linear(cond_dim, embed_dim)
        
    def forward(self, x, cond):
        # x: (seq_len, batch, embed_dim) or (batch, seq_len, embed_dim)
        # cond: (batch, cond_dim)
        normed = self.norm(x)
        scale = self.scale_proj(cond)
        shift = self.shift_proj(cond)
        
        if x.dim() == 3 and x.shape[0] != cond.shape[0]:
            # Handle (seq_len, batch, embed_dim) case
            scale = scale.unsqueeze(0)
            shift = shift.unsqueeze(0)
        else:
            # Handle (batch, seq_len, embed_dim) case
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
            
        return normed * (1 + scale) + shift


class PositionTypeEmbedding(nn.Module):
    """Embedding for position types (framework vs variable)"""
    def __init__(self, embed_dim):
        super().__init__()
        self.framework_emb = nn.Parameter(torch.randn(embed_dim))
        self.variable_emb = nn.Parameter(torch.randn(embed_dim))
        
    def forward(self, position_mask):
        # position_mask: (batch, seq_len) - 0 for framework, 1 for variable
        batch_size, seq_len = position_mask.shape
        embed = torch.zeros(batch_size, seq_len, self.framework_emb.shape[0], device=position_mask.device)
        
        # Framework positions
        framework_mask = (position_mask == 0)
        embed[framework_mask] = self.framework_emb
        
        # Variable positions
        variable_mask = (position_mask == 1)
        embed[variable_mask] = self.variable_emb
        
        return embed


class CategoricalDiffusionTransformer(nn.Module):
    """Transformer for categorical diffusion on protein sequences"""
    
    def __init__(self, seq_len=146, num_classes=21, time_dim=128, embed_dim=128,
                 d_model=32, nhead=8, dim_feedforward=256, dropout=0.1, num_layers=6):
        super().__init__()
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Current sequence embedding (like DiffAb)
        self.current_seq_embedding = nn.Embedding(num_classes + 1, embed_dim)  # +1 for padding
        
        # Time embedding
        self.time_embed = ImprovedTimeEmbedding(time_dim)
        
        # Position type embedding
        self.position_type_embed = PositionTypeEmbedding(embed_dim)
        
        # Relative positional embedding
        self.pe_max_l = 24
        self.embed_pos = nn.Embedding(self.pe_max_l*2+1, d_model)
        
        # Input mixer
        self.input_mixer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        # Transformer blocks with conditional normalization
        self.blocks = nn.ModuleList([
            EnhancedTransformerBlock(
                embed_dim=embed_dim,
                time_dim=time_dim,
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Output projection to amino acid probabilities
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim + time_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_classes)
        )
    
    def relpos(self, seq_len, batch_size, device):
        """Relative positional encodings"""
        p = torch.arange(0, seq_len, device=device)       
        p = p[None,:] - p[:,None]
        bins = torch.linspace(-self.pe_max_l, self.pe_max_l, self.pe_max_l*2+1, device=device)
        b = torch.argmin(torch.abs(bins.view(1, 1, -1) - p.view(p.shape[0], p.shape[1], 1)), dim=-1)
        
        p = self.embed_pos(b)
        p = p.repeat(batch_size, 1, 1, 1)
        return p
    
    def forward(self, x_t, t, position_mask, max_t):
        """
        x_t: (batch, seq_len) - noisy integer sequences
        t: (batch,) - timesteps
        position_mask: (batch, seq_len) - 0 for framework, 1 for variable
        max_t: maximum timestep value
        """
        batch_size, seq_len = x_t.shape
        device = x_t.device
        
        # Get embeddings
        seq_emb = self.current_seq_embedding(x_t)  # (batch, seq_len, embed_dim)
        pos_type_emb = self.position_type_embed(position_mask)  # (batch, seq_len, embed_dim)
        
        # Mix sequence and position type embeddings
        h = self.input_mixer(torch.cat([seq_emb, pos_type_emb], dim=-1))
        
        # Get time embedding
        t_emb = self.time_embed(t, max_t)  # (batch, time_dim)
        
        # Get relative positional embeddings
        rel_p = self.relpos(seq_len, batch_size, device)
        
        # Reshape for transformer: (seq_len, batch, embed_dim)
        h = h.transpose(0, 1)
        
        # Apply transformer blocks
        for block in self.blocks:
            h = block(h, t_emb, rel_p)
        
        # Reshape back: (batch, seq_len, embed_dim)
        h = h.transpose(0, 1)
        
        # Expand time embedding to all positions
        t_emb_expanded = t_emb.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Output projection
        h_with_time = torch.cat([h, t_emb_expanded], dim=-1)
        logits = self.output_proj(h_with_time)  # (batch, seq_len, num_classes)
        
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        
        return probs


class EnhancedTransformerBlock(nn.Module):
    """Transformer block with conditional normalization"""
    def __init__(self, embed_dim, time_dim, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        
        # Attention layer
        self.attn = Layer(in_dim=embed_dim, d_model=d_model, nhead=nhead)
        
        # Feedforward
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim)
        )
        
        # Conditional layer norms
        self.norm1 = ConditionalLayerNorm(embed_dim, time_dim)
        self.norm2 = ConditionalLayerNorm(embed_dim, time_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, t_emb, rel_p):
        # Self-attention with residual
        x_attn = self.attn(x, rel_p)
        x = x + self.dropout(x_attn)
        x = self.norm1(x, t_emb)
        
        # Feedforward with residual
        x_ff = self.ff(x)
        x = x + x_ff
        x = self.norm2(x, t_emb)
        
        return x


def compute_position_mask(sequences_onehot, threshold=0.2):
    """
    Compute position mask: 0 for framework (conserved), 1 for variable
    """
    # Compute entropy at each position
    avg_dist = sequences_onehot.mean(dim=0)  # (seq_len, num_classes)
    entropy = -(avg_dist * torch.log(avg_dist + 1e-8)).sum(dim=-1)
    max_entropy = torch.log(torch.tensor(21.0))
    variability = entropy / max_entropy
    
    # Create mask: 1 for variable positions, 0 for framework
    position_mask = (variability > threshold).float()
    
    return position_mask, variability


def validate_model(model, val_dataloader, categorical_transition, position_mask, num_steps, device):
    """Validation function"""
    model.eval()
    total_val_loss = 0.0
    num_val_batches = 0
    
    with torch.no_grad():
        for x_0, in val_dataloader:
            batch_size_actual = x_0.shape[0]
            
            t = torch.randint(0, num_steps, (batch_size_actual,), device=device)
            
            x_t = categorical_transition.add_noise(x_0, t, device)
            
            position_mask_batch = position_mask.unsqueeze(0).expand(batch_size_actual, -1)
            
            pred_probs = model(x_t, t, position_mask_batch, num_steps)
            
            posterior_true = categorical_transition.posterior(x_t, x_0, t, device)
            
            log_pred_probs = torch.log(pred_probs + 1e-8)
            kl_div = F.kl_div(log_pred_probs, posterior_true, reduction='none').sum(dim=-1)
            
            position_weights = 1 - 0.7 * position_mask_batch
            weighted_kl = kl_div * position_weights
            
            val_loss = weighted_kl.mean()
            total_val_loss += val_loss.item()
            num_val_batches += 1
    
    model.train()
    return total_val_loss / num_val_batches


@torch.no_grad()
def sample_sequences(model, categorical_transition, n_samples, seq_len, num_classes, 
                    position_mask, num_steps, device, temperature=1.0):
    """Generate sequences using categorical diffusion"""
    
    # Start with random sequences
    x_t = torch.randint(0, num_classes, (n_samples, seq_len), device=device)
    
    # Expand position mask for batch
    position_mask_batch = position_mask.unsqueeze(0).expand(n_samples, -1)
    
    for t in tqdm(reversed(range(num_steps)), desc="Sampling"):
        t_batch = torch.full((n_samples,), t, device=device)
        
        # Predict denoised probabilities
        pred_probs = model(x_t, t_batch, position_mask_batch, num_steps)
        
        x_t = categorical_transition.denoise(x_t, pred_probs, t_batch, temperature)
    
    return x_t


def save_model(model, categorical_transition, optimizer, epoch, train_loss, val_loss, train_loss_history, val_loss_history, save_dir="./models"):
    """Save model checkpoint"""
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(save_dir, f"categorical_diffusion_model_{timestamp}.pt")
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_loss_history': train_loss_history,
        'val_loss_history': val_loss_history,
        'model_config': {
            'seq_len': model.seq_len,
            'num_classes': model.num_classes,
            'time_dim': model.time_embed.dim,
            'embed_dim': model.embed_dim,
            'd_model': model.blocks[0].attn.d_model,
            'nhead': model.blocks[0].attn.nhead,
            'dim_feedforward': model.blocks[0].ff[0].out_features,
            'dropout': model.blocks[0].dropout.p,
            'num_layers': len(model.blocks)
        },
        'diffusion_config': {
            'num_steps': categorical_transition.num_steps,
            'num_classes': categorical_transition.num_classes
        }
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Model saved to: {checkpoint_path}")
    return checkpoint_path


def load_model(checkpoint_path, device='cuda'):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Recreate model with saved config
    config = checkpoint['model_config']
    model = CategoricalDiffusionTransformer(
        seq_len=config['seq_len'],
        num_classes=config['num_classes'],
        time_dim=config['time_dim'],
        embed_dim=config['embed_dim'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        num_layers=config['num_layers']
    ).to(device)
    
    # Recreate categorical transition
    diff_config = checkpoint['diffusion_config']
    categorical_transition = CategoricalTransition(
        num_steps=diff_config['num_steps'],
        num_classes=diff_config['num_classes']
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Recreate optimizer if needed
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Model loaded from: {checkpoint_path}")
    print(f"Training was at epoch: {checkpoint['epoch']}")
    #print(f"Last train loss: {checkpoint['train_loss']:.4f}")
    #print(f"Last val loss: {checkpoint['val_loss']:.4f}")
    
    return model, categorical_transition, optimizer, checkpoint['epoch']

def generate_with_saved_model(model_path, position_mask, n_samples=10, temperature=0.8):
    """Generate sequences using a saved model"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load the model
    model, categorical_transition, _, _ = load_model(model_path, device)
    model.eval()
    
    # Generate samples
    with torch.no_grad():
        samples = sample_sequences(
            model, categorical_transition, n_samples=n_samples,
            seq_len=model.seq_len, num_classes=21,
            position_mask=position_mask, num_steps=categorical_transition.num_steps,
            device=device, temperature=temperature
        )
    
    return samples


def save_loss_history(train_loss_history, val_loss_history, save_dir="./visual"):
    """Save loss history to a text file with both train and validation losses"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Use a fixed filename that gets updated each epoch
    loss_file_path = os.path.join(save_dir, "training_loss_history.txt")
    
    with open(loss_file_path, 'w') as f:
        f.write("# Training Loss History\n")
        f.write("# Epoch\tTrain_Loss\tVal_Loss\n")
        for epoch, (train_loss, val_loss) in enumerate(zip(train_loss_history, val_loss_history), 1):
            f.write(f"{epoch}\t{train_loss:.6f}\t{val_loss:.6f}\n")
    
    return loss_file_path


def save_loss_plot(train_loss_history, val_loss_history, save_dir="./visual"):
    """Save loss plot to file with both train and validation curves"""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, linewidth=2, color='blue', label='Train Loss')
    plt.plot(val_loss_history, linewidth=2, color='red', label='Validation Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Categorical Diffusion Training Loss', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(save_dir, f"loss_plot_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Loss plot saved to: {plot_path}")
    
    # Also save as PDF for publication quality
    pdf_path = os.path.join(save_dir, f"loss_plot_{timestamp}.pdf")
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Loss plot (PDF) saved to: {pdf_path}")
    
    plt.show()
    return plot_path


def main(resume_from_checkpoint=None):
    # Parameters
    load_dotenv()
    dat_path = os.getenv('DATA_PATH')
    file_name = os.getenv('FILE_NAME')
    if dat_path is None:
        raise ValueError("DATA_PATH environment variable is required")
    if file_name is None:
        raise ValueError("FILE_NAME environment variable is required")
    
    # Diffusion parameters
    num_steps = 500
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 64
    epochs = 50  
    lr = 1e-3
    max_seq_len = 146
    
    # Model parameters (only used if NOT resuming)
    embed_dim = 128
    d_model = 64
    nhead = 8
    dim_feedforward = 512
    dropout = 0.1
    num_layers = 6
    time_dim = 128
    
    print(f"Using device: {device}")
    
    # Load data
    seq_list, _ = GetFasta(dat_path + '\\' + file_name + '.fasta')
    print(f"Loaded {len(seq_list)} protein sequences")
    
    # Convert to one-hot for analysis
    OHot = GetOneHot(seq_list, max_seq_len, device=device)
    
    # Convert sequences to integers
    from src.Fasta import one_to_num
    sequences_int = []
    for seq in seq_list:
        seq_int = [one_to_num[aa] for aa in seq]
        sequences_int.append(seq_int)
    sequences_int = torch.tensor(sequences_int, device=device)
    
    # Compute position mask
    position_mask, variability = compute_position_mask(OHot, threshold=0.2)
    print(f"Framework positions: {(position_mask == 0).sum().item()}")
    print(f"Variable positions: {(position_mask == 1).sum().item()}")
    
    # Visualize position types and save
    os.makedirs("./visual", exist_ok=True)
    plt.figure(figsize=(12, 4))
    plt.plot(variability.cpu().numpy())
    plt.axhline(y=0.2, color='r', linestyle='--', label='Threshold')
    plt.xlabel('Position')
    plt.ylabel('Variability')
    plt.title('Position Variability in Protein Sequences')
    plt.legend()
    plt.show()
    
    # Create dataset and split into train/validation
    dataset = TensorDataset(sequences_int)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train samples: {train_size}, Validation samples: {val_size}")
    
    # Initialize or load model
    if resume_from_checkpoint is not None:
        print(f"\n{'='*60}")
        print(f"RESUMING TRAINING FROM CHECKPOINT")
        print(f"{'='*60}")
        
        model, categorical_transition, optimizer, start_epoch = load_model(resume_from_checkpoint, device)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, 
            last_epoch=start_epoch
        )
        
        checkpoint = torch.load(resume_from_checkpoint, map_location=device)
        train_loss_history = checkpoint.get('train_loss_history', [])
        val_loss_history = checkpoint.get('val_loss_history', [])
        
        print(f"Resuming from epoch {start_epoch + 1}")
        print(f"Will train for {epochs} more epochs")
        
        start_epoch += 1
        
    else:
        print(f"\n{'='*60}")
        print(f"STARTING FRESH TRAINING")
        print(f"{'='*60}")
        
        categorical_transition = CategoricalTransition(num_steps, num_classes=21)
        
        model = CategoricalDiffusionTransformer(
            seq_len=max_seq_len,
            num_classes=21,
            time_dim=time_dim,
            embed_dim=embed_dim,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_layers=num_layers
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        train_loss_history = []
        val_loss_history = []
        start_epoch = 0
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Starting training...")
    
    for epoch in range(start_epoch, start_epoch + epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        model.train()
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{start_epoch + epochs}")
        
        for batch_idx, (x_0,) in enumerate(pbar):
            batch_size_actual = x_0.shape[0]
            
            # Sample random timesteps
            t = torch.randint(0, num_steps, (batch_size_actual,), device=device)
            
            # Add noise
            x_t = categorical_transition.add_noise(x_0, t, device)
            
            # Expand position mask for batch
            position_mask_batch = position_mask.unsqueeze(0).expand(batch_size_actual, -1)
            
            # Forward pass
            pred_probs = model(x_t, t, position_mask_batch, num_steps)
            
            posterior_true = categorical_transition.posterior(x_t, x_0, t, device)
            
            # KL divergence loss
            log_pred_probs = torch.log(pred_probs + 1e-8)
            kl_div = F.kl_div(log_pred_probs, posterior_true, reduction='none').sum(dim=-1)
            
            # Weight by position mask
            position_weights = 1 - 0.7 * position_mask_batch
            weighted_kl = kl_div * position_weights
            
            loss = weighted_kl.mean()
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=loss.item())
        
        # Validation
        val_loss = validate_model(model, val_dataloader, categorical_transition, position_mask, num_steps, device)
        
        scheduler.step()
        avg_train_loss = epoch_loss / num_batches
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(val_loss)
        
        loss_file_path = save_loss_history(train_loss_history, val_loss_history, save_dir="./visual")
        
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = save_model(
                model, categorical_transition, optimizer, epoch, avg_train_loss, val_loss,
                train_loss_history, val_loss_history, save_dir="./checkpoints"
            )
            print(f"Checkpoint saved at epoch {epoch+1}")
        
        # Generate samples every epoch
        if True:
            model.eval()
            print("\nGenerating samples...")
            
            with torch.no_grad():
                samples = sample_sequences(
                    model, categorical_transition, n_samples=4, 
                    seq_len=max_seq_len, num_classes=21,
                    position_mask=position_mask, num_steps=num_steps,
                    device=device, temperature=0.8
                )
                
                from src.Fasta import num_to_one
                for i in range(min(4, samples.shape[0])):
                    seq = ''.join([num_to_one[idx.item()] for idx in samples[i]])
                    print(f"Sample {i+1}: {seq}")
            
            model.train()
            print()
    
    # Save final model
    final_model_path = save_model(
        model, categorical_transition, optimizer, start_epoch + epochs - 1, 
        train_loss_history[-1], val_loss_history[-1], train_loss_history, val_loss_history, 
        save_dir="./final_models"
    )
    
    # Save loss plot
    loss_plot_path = save_loss_plot(train_loss_history, val_loss_history, save_dir="./visual")
    
    # Save final loss history
    final_loss_file_path = save_loss_history(train_loss_history, val_loss_history, save_dir="./visual")
    
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Total epochs trained: {len(train_loss_history)}")
    print(f"Final train loss: {train_loss_history[-1]:.6f}")
    print(f"Final val loss: {val_loss_history[-1]:.6f}")


if __name__ == "__main__":
    # Fresh training
    #main()
    
    # To resume training from a checkpoint, uncomment and modify path:
    main(resume_from_checkpoint="./checkpoints/categorical_diffusion_model_20250721_005827.pt")
