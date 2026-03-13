# -----------------------------------------------------------------------------
# Conditional Variational Autoencoder (CVAE) with Concat conditioning
# For learning p(Δq_contact | q_0, x_idx)
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np

class CVAE(nn.Module):
    """
    Conditional VAE with concat conditioning
    
    Encoder: [Δq, c] -> z ~ N(μ, σ²)
    Decoder: [z, c] -> Δq_recon
    
    where c = [q_0, x_idx] is the condition
    """
    def __init__(self, data_dim, condition_dim, latent_dim, hidden_dims=[256, 128]):
        """
        Args:
            data_dim: dimension of Δq (num_joints)
            condition_dim: dimension of condition c (num_joints + 2)
            latent_dim: dimension of latent space z
            hidden_dims: list of hidden layer dimensions
        """
        super(CVAE, self).__init__()
        
        self.data_dim = data_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        
        # Encoder: [Δq, c] -> μ, logσ²
        encoder_layers = []
        input_dim = data_dim + condition_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(input_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim)
            ])
            input_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder: [z, c] -> Δq
        decoder_layers = []
        input_dim = latent_dim + condition_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(input_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim)
            ])
            input_dim = h_dim
        decoder_layers.append(nn.Linear(hidden_dims[0], data_dim))
        # No activation on output (regression task)
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, delta_q, condition):
        """
        Encode [Δq, c] to latent distribution parameters
        
        Args:
            delta_q: (B, data_dim)
            condition: (B, condition_dim)
        Returns:
            mu: (B, latent_dim)
            logvar: (B, latent_dim)
        """
        x = torch.cat([delta_q, condition], dim=-1)
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = μ + σ * ε, ε ~ N(0, I)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, condition):
        """
        Decode [z, c] to Δq
        
        Args:
            z: (B, latent_dim)
            condition: (B, condition_dim)
        Returns:
            delta_q_recon: (B, data_dim)
        """
        x = torch.cat([z, condition], dim=-1)
        delta_q_recon = self.decoder(x)
        return delta_q_recon

    def forward(self, delta_q, condition):
        """
        Forward pass through CVAE
        
        Args:
            delta_q: (B, data_dim) - target Δq
            condition: (B, condition_dim) - [q_0, x_idx]
        Returns:
            delta_q_recon: (B, data_dim)
            mu: (B, latent_dim)
            logvar: (B, latent_dim)
        """
        mu, logvar = self.encode(delta_q, condition)
        z = self.reparameterize(mu, logvar)
        delta_q_recon = self.decode(z, condition)
        return delta_q_recon, mu, logvar
    
    def sample(self, condition, n_samples=1):
        """
        Sample from the model given condition
        
        Args:
            condition: (B, condition_dim) or (condition_dim,)
            n_samples: number of samples per condition
        Returns:
            delta_q_samples: (B, n_samples, data_dim) or (n_samples, data_dim)
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
        z_flat = z.reshape(-1, self.latent_dim)
        condition_flat = condition_expanded.reshape(-1, self.condition_dim)
        delta_q_flat = self.decode(z_flat, condition_flat)
        delta_q_samples = delta_q_flat.reshape(batch_size, n_samples, self.data_dim)
        
        if squeeze_output:
            delta_q_samples = delta_q_samples.squeeze(0)
        return delta_q_samples
    
    def loss_function(self, delta_q_recon, delta_q, mu, logvar, beta=1.0):
        """
        CVAE loss = Reconstruction loss + β * KL divergence
        
        Args:
            delta_q_recon: (B, data_dim) - reconstructed Δq
            delta_q: (B, data_dim) - ground truth Δq
            mu: (B, latent_dim)
            logvar: (B, latent_dim)
            beta: weight for KL term (default: 1.0)
                increase beta to enforce more regularization
        Returns:
            loss: scalar
            recon_loss: scalar
            kl_loss: scalar
        """
        # Reconstruction loss (MSE for regression)
        recon_loss = nn.functional.mse_loss(delta_q_recon, delta_q, reduction='mean')
        
        # KL divergence: KL(N(μ, σ²) || N(0, I))
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        loss = recon_loss + beta * kl_loss
        
        return loss, recon_loss, kl_loss

    def grad_wrt_condition(self, condition, z=None, delta_q_target=None):
        """
        Compute gradient of decoded output w.r.t. condition
        
        Two modes:
        1. If z is provided: compute ∂Decoder(z, c)/∂c (direct sensitivity)
        2. If delta_q_target provided: compute gradient through full forward pass
        
        Args:
            condition: (B, condition_dim) - [q_0, x_idx], requires_grad=True
            z: (B, latent_dim) - optional, latent code
            delta_q_target: (B, data_dim) - optional, for full forward pass
        Returns:
            grad: (B, condition_dim)
        """
        
        # Save original training state
        was_training = self.training
        self.train()  # Enable gradient flow
        condition = condition.clone().detach().requires_grad_(True)
        
        if z is not None:
            # Mode 1: Direct decoder gradient (recommended for vector field)
            # This computes how decoder output changes with condition given fixed z
            delta_q_recon = self.decode(z, condition)
            
            # Sum over data_dim to get scalar, then compute gradient
            output_mean = delta_q_recon.mean()
            grad = torch.autograd.grad(
                outputs=output_mean,
                inputs=condition,
                create_graph=False,
                retain_graph=False
            )[0]
            
        elif delta_q_target is not None:
            # Mode 2: Full forward pass gradient
            # This is more complex and includes encoder influence
            mu, logvar = self.encode(delta_q_target, condition)
            z_sampled = self.reparameterize(mu, logvar)
            delta_q_recon = self.decode(z_sampled, condition)
            
            output_mean = delta_q_recon.mean()
            grad = torch.autograd.grad(
                outputs=output_mean,
                inputs=condition,
                create_graph=False,
                retain_graph=False,
                allow_unused=True
            )[0]
        else:
            raise ValueError("Either z or delta_q_target must be provided")
        
        # Restore original training state
        self.train(was_training)
        
        return grad
    
    def compute_vector_field(self, condition, n_samples=10):
        """
        Compute average gradient direction for vector field visualization
        
        Args:
            condition: (B, condition_dim) or (condition_dim,)
            n_samples: number of z samples to average over
        Returns:
            avg_grad: (B, condition_dim) or (condition_dim,) - average gradient direction
        """
        if condition.dim() == 1:
            condition = condition.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = condition.shape[0]
        condition = condition.clone().detach().requires_grad_(True)
        # check if condition requires grad
        assert condition.requires_grad, "Condition tensor must require gradients"
        # Sample multiple z from prior
        z_samples = torch.randn(batch_size, n_samples, self.latent_dim).to(condition.device)
        
        gradients = []
        distances = []  
        for i in range(n_samples):
            z = z_samples[:, i, :]  # (B, latent_dim)
            # Decode
            delta_q = self.decode(z, condition)
            # Compute gradient
            output_mean = delta_q.mean()

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
        grad_mean = torch.norm(gradients, dim=0, keepdim=True)
        grad_var = torch.var(gradients, dim=0, keepdim=True)
        if squeeze_output:
            avg_grad = avg_grad.squeeze(0)
        # avg_grad: (B, condition_dim) or (condition_dim,)
        # grad_mean: (B, condition_dim)
        # grad_var: (B, condition_dim)
        # distances: (n_samples, B)
        return avg_grad, grad_mean.squeeze(0), grad_var.squeeze(0), torch.stack(distances, dim=0).mean(dim=0)