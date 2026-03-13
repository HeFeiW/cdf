# -----------------------------------------------------------------------------
# Transformer-based Conditional Variational Autoencoder (TCVAE)
# For learning p(d | q_0, x_idx) where d is distance feature
# Uses Transformer architecture instead of fully-connected layers
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, embed_dim, max_len=100, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (B, seq_len, embed_dim)
        Returns:
            x + positional encoding: (B, seq_len, embed_dim)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TCVAE(nn.Module):
    """
    Transformer-based Conditional VAE
    
    Encoder: [d, c] -> Transformer Encoder -> μ, σ²
    Decoder: [z, c] -> Transformer Decoder -> d_recon
    
    where:
        d: distance feature (data_dim=1)
        c: condition [q_0, x_idx] (condition_dim=num_joints+2)
        z: latent code (latent_dim)
    """
    def __init__(
        self, 
        data_dim, 
        condition_dim, 
        latent_dim, 
        embed_dim=128,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        ff_dim=512,
        dropout=0.1,
        use_positional_encoding=True
    ):
        """
        Args:
            data_dim: dimension of d (typically 1 for distance)
            condition_dim: dimension of condition c (num_joints + 2)
            latent_dim: dimension of latent space z
            embed_dim: embedding dimension for transformer
            num_encoder_layers: number of transformer encoder layers
            num_decoder_layers: number of transformer decoder layers
            num_heads: number of attention heads
            ff_dim: feedforward network dimension
            dropout: dropout rate
            use_positional_encoding: whether to use positional encoding
        """
        super(TCVAE, self).__init__()
        
        self.data_dim = data_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.use_positional_encoding = use_positional_encoding
        
        # Sequence length: 1 (data) + condition_dim (each dimension as a token)
        self.seq_len_encoder = 1 + condition_dim
        self.seq_len_decoder = 1 + condition_dim  # z token + condition tokens
        
        # ===================== Encoder Components =====================
        
        # Input embeddings for encoder
        self.data_embed = nn.Linear(1, embed_dim)  # Each dim of data -> embed_dim
        self.condition_embed = nn.Linear(1, embed_dim)  # Each dim of condition -> embed_dim
        
        # Positional encoding
        if use_positional_encoding:
            self.pos_encoder = PositionalEncoding(embed_dim, max_len=100, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,  # Input shape: (B, seq_len, embed_dim)
            norm_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(embed_dim)
        )
        
        # Latent projection heads (from aggregated encoding)
        self.fc_mu = nn.Linear(embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(embed_dim, latent_dim)
        
        # ===================== Decoder Components =====================
        
        # Latent embedding
        self.latent_embed = nn.Linear(latent_dim, embed_dim)
        
        # Positional encoding for decoder (shared or separate)
        if use_positional_encoding:
            self.pos_decoder = PositionalEncoding(embed_dim, max_len=100, dropout=dropout)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(embed_dim)
        )
        
        # Output projection (only decode the first token -> distance)
        self.output_proj = nn.Linear(embed_dim, data_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _embed_condition(self, condition):
        """
        Embed condition into token sequence
        
        Args:
            condition: (B, condition_dim)
        Returns:
            condition_tokens: (B, condition_dim, embed_dim)
        """
        batch_size = condition.shape[0]
        
        # Each dimension becomes a token
        # condition: (B, condition_dim) -> (B, condition_dim, 1) -> (B, condition_dim, embed_dim)
        condition_tokens = self.condition_embed(condition.unsqueeze(-1))
        
        return condition_tokens
    
    def encode(self, data, condition):
        """
        Encode [d, c] to latent distribution parameters
        
        Args:
            data: (B, data_dim) - distance feature
            condition: (B, condition_dim) - [q_0, x_idx]
        Returns:
            mu: (B, latent_dim)
            logvar: (B, latent_dim)
        """
        batch_size = data.shape[0]
        
        # Embed data: (B, data_dim) -> (B, 1, embed_dim)
        data_token = self.data_embed(data.unsqueeze(1))  # (B, 1, embed_dim)
        
        # Embed condition: (B, condition_dim) -> (B, condition_dim, embed_dim)
        condition_tokens = self._embed_condition(condition)
        
        # Concatenate into sequence: (B, 1+condition_dim, embed_dim)
        encoder_input = torch.cat([data_token, condition_tokens], dim=1)
        
        # Add positional encoding
        if self.use_positional_encoding:
            encoder_input = self.pos_encoder(encoder_input)
        
        # Pass through transformer encoder
        encoder_output = self.transformer_encoder(encoder_input)  # (B, seq_len, embed_dim)
        
        # Aggregate: use mean pooling over sequence
        aggregated = encoder_output.mean(dim=1)  # (B, embed_dim)
        
        # Project to latent distribution parameters
        mu = self.fc_mu(aggregated)  # (B, latent_dim)
        logvar = self.fc_logvar(aggregated)  # (B, latent_dim)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = μ + σ * ε, ε ~ N(0, I)
        
        Args:
            mu: (B, latent_dim)
            logvar: (B, latent_dim)
        Returns:
            z: (B, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, condition):
        """
        Decode [z, c] to reconstructed distance
        
        Args:
            z: (B, latent_dim)
            condition: (B, condition_dim)
        Returns:
            data_recon: (B, data_dim)
        """
        batch_size = z.shape[0]
        
        # Embed latent code as query token: (B, latent_dim) -> (B, 1, embed_dim)
        z_token = self.latent_embed(z).unsqueeze(1)  # (B, 1, embed_dim)
        
        # Embed condition as memory: (B, condition_dim) -> (B, condition_dim, embed_dim)
        condition_tokens = self._embed_condition(condition)
        
        # Add positional encoding to both
        if self.use_positional_encoding:
            z_token = self.pos_decoder(z_token)
            condition_tokens = self.pos_decoder(condition_tokens)
        
        # Transformer decoder
        # tgt: (B, 1, embed_dim) - what we want to decode
        # memory: (B, condition_dim, embed_dim) - conditioning information
        decoder_output = self.transformer_decoder(
            tgt=z_token,
            memory=condition_tokens
        )  # (B, 1, embed_dim)
        
        # Project to data space (only use the first token)
        data_recon = self.output_proj(decoder_output.squeeze(1))  # (B, data_dim)
        
        return data_recon
    
    def forward(self, data, condition):
        """
        Forward pass through TCVAE
        
        Args:
            data: (B, data_dim) - target distance
            condition: (B, condition_dim) - [q_0, x_idx]
        Returns:
            data_recon: (B, data_dim)
            mu: (B, latent_dim)
            logvar: (B, latent_dim)
        """
        mu, logvar = self.encode(data, condition)
        z = self.reparameterize(mu, logvar)
        data_recon = self.decode(z, condition)
        return data_recon, mu, logvar
    
    def sample(self, condition, n_samples=1):
        """
        Sample from the model given condition
        
        Args:
            condition: (B, condition_dim) or (condition_dim,)
            n_samples: number of samples per condition
        Returns:
            data_samples: (B, n_samples, data_dim) or (n_samples, data_dim)
        """
        if condition.dim() == 1:
            condition = condition.unsqueeze(0)  # (1, condition_dim)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = condition.shape[0]
        
        # Sample z ~ N(0, I)
        z = torch.randn(batch_size, n_samples, self.latent_dim).to(condition.device)
        
        # Expand condition for all samples
        condition_expanded = condition.unsqueeze(1).expand(-1, n_samples, -1)
        
        # Decode
        z_flat = z.reshape(-1, self.latent_dim)  # (B*n_samples, latent_dim)
        condition_flat = condition_expanded.reshape(-1, self.condition_dim)  # (B*n_samples, condition_dim)
        data_flat = self.decode(z_flat, condition_flat)  # (B*n_samples, data_dim)
        data_samples = data_flat.reshape(batch_size, n_samples, self.data_dim)
        
        if squeeze_output:
            data_samples = data_samples.squeeze(0)  # (n_samples, data_dim)
        
        return data_samples
    
    def loss_function(self, data_recon, data, mu, logvar, beta=1.0):
        """
        TCVAE loss = Reconstruction loss + β * KL divergence
        
        Args:
            data_recon: (B, data_dim) - reconstructed distance
            data: (B, data_dim) - ground truth distance
            mu: (B, latent_dim)
            logvar: (B, latent_dim)
            beta: weight for KL term
        Returns:
            loss: scalar
            recon_loss: scalar
            kl_loss: scalar
        """
        # Reconstruction loss (MSE for regression)
        recon_loss = nn.functional.mse_loss(data_recon, data, reduction='mean')
        
        # KL divergence: KL(N(μ, σ²) || N(0, I))
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        loss = recon_loss + beta * kl_loss
        
        return loss, recon_loss, kl_loss
    
    def compute_vector_field(self, condition, n_samples=10):
        """
        Compute average gradient direction for vector field visualization
        
        Args:
            condition: (B, condition_dim) or (condition_dim,)
            n_samples: number of z samples to average over
        Returns:
            avg_grad: (B, condition_dim) - average gradient
            grad_mean: (B, condition_dim) - mean of decoded outputs
            grad_var: (B, condition_dim) - variance of decoded outputs
            distances: average distance prediction
        """
        if condition.dim() == 1:
            condition = condition.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = condition.shape[0]
        condition = condition.clone().detach().requires_grad_(True)
        
        # Sample multiple z from prior
        z_samples = torch.randn(batch_size, n_samples, self.latent_dim).to(condition.device)
        
        gradients = []
        distances = []
        
        for i in range(n_samples):
            z = z_samples[:, i, :]  # (B, latent_dim)
            
            # Decode
            data = self.decode(z, condition)  # (B, data_dim)
            
            # Compute gradient w.r.t. condition
            output_mean = data.mean()
            grad = torch.autograd.grad(
                outputs=output_mean,
                inputs=condition,
                create_graph=False,
                retain_graph=True
            )[0]
            
            gradients.append(grad)
            distances.append(output_mean.detach())
        
        # Average over samples
        avg_grad = torch.stack(gradients, dim=0).mean(dim=0)  # (B, condition_dim)
        gradients = torch.stack(gradients, dim=0)  # (n_samples, B, condition_dim)
        grad_mean = torch.norm(gradients, dim=0, keepdim=True)  # (1, B, condition_dim)
        grad_var = torch.var(gradients, dim=0, keepdim=True)  # (1, B, condition_dim)
        
        if squeeze_output:
            avg_grad = avg_grad.squeeze(0)
        
        return (
            avg_grad,
            grad_mean.squeeze(0),
            grad_var.squeeze(0),
            torch.stack(distances, dim=0).mean(dim=0)
        )


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model parameters
    data_dim = 1
    condition_dim = 4  # 2 joints + 2 workspace dims
    latent_dim = 32
    batch_size = 16
    
    # Create model
    model = TCVAE(
        data_dim=data_dim,
        condition_dim=condition_dim,
        latent_dim=latent_dim,
        embed_dim=128,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        ff_dim=512,
        dropout=0.1
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    data = torch.randn(batch_size, data_dim).to(device)
    condition = torch.randn(batch_size, condition_dim).to(device)
    
    data_recon, mu, logvar = model(data, condition)
    loss, recon_loss, kl_loss = model.loss_function(data_recon, data, mu, logvar)
    
    print(f"\nForward pass test:")
    print(f"  Input data shape: {data.shape}")
    print(f"  Input condition shape: {condition.shape}")
    print(f"  Reconstructed data shape: {data_recon.shape}")
    print(f"  Loss: {loss.item():.4f}, Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f}")
    
    # Test sampling
    samples = model.sample(condition[0], n_samples=10)
    print(f"\nSampling test:")
    print(f"  Samples shape: {samples.shape}")
    
    print("\n✓ TCVAE implementation test passed!")