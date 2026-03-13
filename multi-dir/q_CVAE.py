# -----------------------------------------------------------------------------
# Training and Testing Script for Contact Configuration CVAE
# Learning p(Δq_contact | q_0, x_idx) using concat conditioning
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
from CVAE import CVAE
from TCVAE import TCVAE
from cdf_dataloader import get_dataloader
import sys
sys.path.append('../2Dexamples')
from cdf import CDF2D
from primitives2D_torch import Circle


def train_cvae(model, dataloader, optimizer, device, epoch, beta=1.0):
    """
    Train CVAE for one epoch
    
    Args:
        model: CVAE model
        dataloader: training data loader
        optimizer: optimizer
        device: cuda or cpu
        epoch: current epoch number
        beta: weight for KL term
    
    Returns:
        avg_loss, avg_recon_loss, avg_kl_loss
    """
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for condition, delta_q in pbar:
        condition = condition.to(device)
        delta_q = delta_q.to(device)
        d = torch.norm(delta_q, dim=-1).unsqueeze(-1)  # (B, 1, 1)
        d_recon, mu, logvar = model(d, condition)
        loss, recon_loss, kl_loss = model.loss_function(
            d_recon, d, mu, logvar, beta=beta
        )
        # Forward pass
        # delta_q_recon, mu, logvar = model(delta_q, condition)
        
        # Compute loss
        # loss, recon_loss, kl_loss = model.loss_function(
        #     delta_q_recon, delta_q, mu, logvar, beta=beta
        # )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'recon': f'{recon_loss.item():.4f}',
            'kl': f'{kl_loss.item():.4f}'
        })
    
    n_batches = len(dataloader)
    return total_loss / n_batches, total_recon_loss / n_batches, total_kl_loss / n_batches


def test_cvae(model, dataloader, device, beta=1.0):
    """
    Test CVAE on validation set
    """
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    with torch.no_grad():
        for condition, delta_q in dataloader:
            condition = condition.to(device)
            delta_q = delta_q.to(device)
            
            # Forward pass
            delta_q_recon, mu, logvar = model(delta_q, condition)
            
            # Compute loss
            loss, recon_loss, kl_loss = model.loss_function(
                delta_q_recon, delta_q, mu, logvar, beta=beta
            )
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
    
    n_batches = len(dataloader)
    return total_loss / n_batches, total_recon_loss / n_batches, total_kl_loss / n_batches


def visualize_samples(model, cdf, x_idx, device, n_samples=50, save_path=None, ax=None):
    """
    Visualize sampled contact configurations for a given workspace point
    
    Args:
        model: trained CVAE model
        cdf: CDF2D instance
        x_idx: workspace point (2,)
        device: cuda or cpu
        n_samples: number of samples to generate
        save_path: path to save figure
    """
    model.eval()
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    axes = [ax]
    
    # Get ground truth contact configurations for this workspace point
    obj = Circle(center=x_idx, radius=0.001, device=device)
    q_gt = cdf.find_q([obj])[1]  # Ground truth contact configs
    
    # Sample multiple q_0
    n_q0 = 5
    q0_samples = torch.rand(n_q0, cdf.num_joints).to(device) * (cdf.q_max - cdf.q_min) + cdf.q_min
    
    colors = plt.cm.rainbow(np.linspace(0, 1, n_q0))
    
    with torch.no_grad():
        for idx, q_0 in enumerate(q0_samples):
            # Construct condition
            condition = torch.cat([q_0, x_idx], dim=0).to(device)
            # Sample from model
            # delta_q_samples = model.sample(condition, n_samples=n_samples)  # (n_samples, num_joints)
            # q_contact_samples = q_0.unsqueeze(0) + delta_q_samples  # (n_samples, num_joints)
            d_samples = model.sample(condition, n_samples=n_samples)  # (n_samples, 1)
            # Plot in configuration space
            ax1 = axes[0]
            # ax1.scatter(
            #     q_contact_samples[:, 0].cpu().numpy(),
            #     q_contact_samples[:, 1].cpu().numpy() if cdf.num_joints > 1 else np.zeros(n_samples),
            #     c=[colors[idx]], alpha=0.6, s=20, label=f'q_0 {idx+1} (sampled)'
            # )
            ax1.scatter(
                q_0[0].cpu().numpy(),
                q_0[1].cpu().numpy() if cdf.num_joints > 1 else 0,
                c=[colors[idx]], marker='*', s=200, edgecolors='black', linewidths=1.5
            )
            for d_contact in d_samples:
                circle = plt.Circle(
                    (q_0[0].cpu().numpy(),
                    q_0[1].cpu().numpy() if cdf.num_joints > 1 else 0),
                    d_contact,
                    color=colors[idx], fill=False, linestyle=':', linewidth=1.0,
                )
                ax1.add_patch(circle)
            # mean_dist, var_dist, mean_contact = mean_distance(delta_q_samples)
            # print(f"q_0 {idx+1}: Mean distance to target object: {mean_dist:.4f}, Variance: {var_dist:.4f}")
            # 用圆圈表示平均距离
            # circle = plt.Circle(
            #     (q_0[0].cpu().numpy(), q_0[1].cpu().numpy() if cdf.num_joints > 1 else 0),
            #     mean_dist, color=colors[idx], fill=False, linestyle='--', linewidth=2,
            #     label=f'q_0 {idx+1} (mean dist)'
            #     )
            # ax1.add_patch(circle)
            # ax1.scatter(
            #     q_0[0].cpu().numpy() + mean_contact[0].cpu().numpy(),
            #     q_0[1].cpu().numpy() + mean_contact[1].cpu().numpy() if cdf.num_joints > 1 else 0,
            #     c=[colors[idx]], marker='D', s=100, edgecolors='black', linewidths=1.5,
            #     label=f'q_0 {idx+1} (mean contact)'
            # )
            # 固定坐标轴范围
            ax1.set_xlim(-np.pi, np.pi)
            ax1.set_ylim(-np.pi, np.pi)
            ax1.set_aspect('equal')
    
    # Plot ground truth contact configurations
    if len(q_gt) > 0:
        ax1.scatter(
            q_gt[:, 0].cpu().numpy(),
            q_gt[:, 1].cpu().numpy() if cdf.num_joints > 1 else np.zeros(len(q_gt)),
            c='black', marker='x', s=50, alpha=0.3, linewidths=2, 
            label='Ground Truth Contact', zorder=10
        )
    
    ax1.set_xlabel('q1', fontsize=12)
    ax1.set_ylabel('q2', fontsize=12)
    ax1.set_title(f'Sampled vs Ground Truth Contact Configs\nfor x = [{x_idx[0]:.2f}, {x_idx[1]:.2f}]', fontsize=12)
    # ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-np.pi, np.pi)
    ax1.set_ylim(-np.pi, np.pi)
    
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()

def visualize_vector_field(model, cdf, x_test, device, n_samples=50, save_path=None, ax=None):
    """
    Visualize the vector field of predicted distances over the workspace
    Args:
        model: trained CVAE model
        cdf: CDF2D instance
        x_test: workspace point (2,)
        device: cuda or cpu
        n_samples: number of samples to generate
        save_path: path to save figure
    """
    model.eval()
    # Create a grid of points in the configuration space
    # sample from cdf.q_min to cdf.q_max
    q1 = torch.linspace(cdf.q_min[0], cdf.q_max[0], steps=20)
    q2 = torch.linspace(cdf.q_min[1], cdf.q_max[1], steps=20)
    Q1, Q2 = torch.meshgrid(q1, q2, indexing='ij')
    Q_grid = torch.stack([Q1.flatten(), Q2.flatten()], dim=-1).to(device)  # (N, 2)
    # For each q_0 in the grid, compute gradient of predicted distance to q_0
    U = torch.zeros(Q_grid.shape).to(device)
    M = torch.zeros(Q_grid.shape).to(device)
    V = torch.zeros(Q_grid.shape).to(device)
    for i, q_0 in enumerate(Q_grid):
        # Construct condition
        condition = torch.cat([q_0, x_test], dim=0).expand(1, -1).to(device)  # (1, condition_dim)
        gradients, mean, var, _ = model.compute_vector_field(condition, n_samples=n_samples)  # (condition_dim,)
        # Extract gradient w.r.t q_0
        grad_q0 = gradients[0, :cdf.num_joints]  # (num_joints,)
        U[i] = -grad_q0  # Negative gradient points towards decreasing distance
        M[i] = mean[0, :cdf.num_joints]
        V[i] = var[0, :cdf.num_joints]
    U = U.cpu().numpy()
    M = M.cpu().numpy()
    V = V.cpu().numpy()
    Q_grid = Q_grid.cpu().numpy()
    # Reshape for quiver plot
    U1 = U[:, 0].reshape(Q1.shape)
    U2 = U[:, 1].reshape(Q2.shape)
    Q1 = Q1.cpu().numpy()
    Q2 = Q2.cpu().numpy()
    # Plot vector field
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    ax.quiver(Q1, Q2, U1, U2, color='blue', alpha=0.6)
    # 在q_0位置用椭圆表示均值和方差
    for i in range(Q_grid.shape[0]):
        mean_vec = M[i]
        var_vec = V[i]
        ellipse = plt.matplotlib.patches.Ellipse(
            (Q_grid[i, 0], Q_grid[i, 1]),
            width=2*np.sqrt(var_vec[0]),
            height=2*np.sqrt(var_vec[1]),
            angle=0,
            edgecolor='red',
            facecolor='none',
            linestyle='--',
            alpha=0.5
        )
        ax.add_patch(ellipse)
    ax.set_xlabel('q1', fontsize=12)
    ax.set_ylabel('q2', fontsize=12)
    ax.set_title('Vector Field of Predicted Distance Gradient', fontsize=12)
    ax.set_xlim(cdf.q_min[0].cpu().numpy(), cdf.q_max[0].cpu().numpy())
    ax.set_ylim(cdf.q_min[1].cpu().numpy(), cdf.q_max[1].cpu().numpy())
    ax.set_aspect('equal')
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved vector field visualization to {save_path}")
    plt.show()
    
def visualize_gradient_magnitude(model, cdf, x_test, device, n_samples=50, save_path=None, ax=None):
    """
    Visualize the vector field of predicted distances over the workspace
    Args:
        model: trained CVAE model
        cdf: CDF2D instance
        x_test: workspace point (2,)
        device: cuda or cpu
        n_samples: number of samples to generate
        save_path: path to save figure
    """
    model.eval()
    # Create a grid of points in the configuration space
    # sample from cdf.q_min to cdf.q_max
    q1 = torch.linspace(cdf.q_min[0], cdf.q_max[0], steps=20)
    q2 = torch.linspace(cdf.q_min[1], cdf.q_max[1], steps=20)
    Q1, Q2 = torch.meshgrid(q1, q2, indexing='ij')
    Q_grid = torch.stack([Q1.flatten(), Q2.flatten()], dim=-1).to(device)  # (N, 2)
    # For each q_0 in the grid, compute gradient of predicted distance to q_0
    U = torch.zeros(Q_grid.shape).to(device)
    M = torch.zeros(Q_grid.shape).to(device)
    V = torch.zeros(Q_grid.shape).to(device)
    for i, q_0 in enumerate(Q_grid):
        # Construct condition
        condition = torch.cat([q_0, x_test], dim=0).expand(1, -1).to(device)  # (1, condition_dim)
        _, mean, _, _ = model.compute_vector_field(condition, n_samples=n_samples)  # (condition_dim,)
        # Extract gradient w.r.t q_0
        M[i] = mean[0, :cdf.num_joints]
    M = M.cpu().numpy()
    magnitude = np.linalg.norm(M, axis=-1)
    print('magnitude shape:',magnitude.shape)
    Q_grid = Q_grid.cpu().numpy()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    sc = ax.contourf(
        Q_grid[:, 0].reshape(Q1.shape),
        Q_grid[:, 1].reshape(Q2.shape),
        np.linalg.norm(M, axis=-1).reshape(Q1.shape),
        levels=20,
        cmap='viridis'
    )
    ax.set_xlabel('q1', fontsize=12)
    ax.set_ylabel('q2', fontsize=12)
    ax.set_title('Vector Field of Predicted Distance Gradient', fontsize=12)
    ax.set_xlim(cdf.q_min[0].cpu().numpy(), cdf.q_max[0].cpu().numpy())
    ax.set_ylim(cdf.q_min[1].cpu().numpy(), cdf.q_max[1].cpu().numpy())
    ax.set_aspect('equal')
    ax.figure.colorbar(sc, ax=ax, label='Gradient Magnitude')
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved vector field visualization to {save_path}")
    plt.show()

def visualize_d_pred(model, cdf, x_test, device, n_samples=50, save_path=None):
    """
    Visualize the vector field of predicted distances over the workspace
    Args:
        model: trained CVAE model
        cdf: CDF2D instance
        x_test: workspace point (2,)
        device: cuda or cpu
        n_samples: number of samples to generate(the d drawn is averaged)
        save_path: path to save figure
    """
    model.eval()
    # Create a grid of points in the configuration space
    # sample from cdf.q_min to cdf.q_max
    q1 = torch.linspace(cdf.q_min[0], cdf.q_max[0], steps=20)
    q2 = torch.linspace(cdf.q_min[1], cdf.q_max[1], steps=20)
    Q1, Q2 = torch.meshgrid(q1, q2, indexing='ij')
    Q_grid = torch.stack([Q1.flatten(), Q2.flatten()], dim=-1).to(device)  # (N, 2)
    # For each q_0 in the grid, compute gradient of predicted distance to q_0
    D = torch.zeros(Q_grid.shape[0], 1).to(device)
    for i, q_0 in enumerate(Q_grid):
        # Construct condition
        condition = torch.cat([q_0, x_test], dim=0).expand(1, -1).to(device)  # (1, condition_dim)
        d = model.sample(condition, n_samples=n_samples)  # (n_samples, 1)
        d = d.squeeze(-1)  # (n_samples,)
        D[i] = d.mean(dim=-1)
    GT_D = torch.zeros_like(D)
    ERR = torch.zeros_like(D)
    for i, q_0 in enumerate(Q_grid):
        GT_D[i] = cdf.calculate_cdf(q_0.unsqueeze(0), [Circle(center=x_test, radius=0.001, device=device)], method='online_computation', return_grad=False) # use offline_grid for speed
        if GT_D[i] == float('inf'):
            GT_D[i] = torch.tensor(0.0).to(device)
        ERR[i] = torch.abs(D[i] - GT_D[i])
    D = D.detach().cpu().numpy()
    Q_grid = Q_grid.cpu().numpy()
    GT_D = GT_D.detach().cpu().numpy()
    ERR = ERR.detach().cpu().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax = axes[0]
    ax1 = axes[1]
    ax2 = axes[2]
    # Plot d_pred
    sc = ax.contourf(
        Q_grid[:, 0].reshape(Q1.shape),
        Q_grid[:, 1].reshape(Q2.shape),
        D.reshape(Q1.shape),
        levels=20,
        cmap='viridis'
    )
    ax.set_xlabel('q1', fontsize=12)
    ax.set_ylabel('q2', fontsize=12)
    ax.set_title('d_pred', fontsize=12)
    ax.set_xlim(cdf.q_min[0].cpu().numpy(), cdf.q_max[0].cpu().numpy())
    ax.set_ylim(cdf.q_min[1].cpu().numpy(), cdf.q_max[1].cpu().numpy())
    ax.set_aspect('equal')
    ax.figure.colorbar(sc, ax=ax, label='Predicted Distance d_pred')
    plt.grid(True, alpha=0.3)
    # Plot GT d
    sc1 = ax1.contourf(
        Q_grid[:, 0].reshape(Q1.shape),
        Q_grid[:, 1].reshape(Q2.shape),
        GT_D.reshape(Q1.shape),
        levels=20,
        cmap='viridis'
    )
    ax1.set_xlabel('q1', fontsize=12)
    ax1.set_ylabel('q2', fontsize=12)
    ax1.set_title('GT d', fontsize=12)
    ax1.set_xlim(cdf.q_min[0].cpu().numpy(), cdf.q_max[0].cpu().numpy())
    ax1.set_ylim(cdf.q_min[1].cpu().numpy(), cdf.q_max[1].cpu().numpy())
    ax1.set_aspect('equal')
    ax1.figure.colorbar(sc1, ax=ax1, label='Ground Truth Distance GT d')
    plt.grid(True, alpha=0.3)
    # Plot Error
    sc2 = ax2.contourf(
        Q_grid[:, 0].reshape(Q1.shape),
        Q_grid[:, 1].reshape(Q2.shape),
        ERR.reshape(Q1.shape),
        levels=20,
        cmap='viridis'
    )
    ax2.set_xlabel('q1', fontsize=12)
    ax2.set_ylabel('q2', fontsize=12)
    ax2.set_title('Error', fontsize=12)
    ax2.set_xlim(cdf.q_min[0].cpu().numpy(), cdf.q_max[0].cpu().numpy())
    ax2.set_ylim(cdf.q_min[1].cpu().numpy(), cdf.q_max[1].cpu().numpy())
    ax2.set_aspect('equal')
    ax2.figure.colorbar(sc2, ax=ax2, label='Absolute Error')
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved vector field visualization to {save_path}")
    plt.show()

def evaluate_accuracy(model, cdf, device, n_test_points=10, n_samples=20):
    """
    Evaluate how well the sampled configurations reach the target workspace points
    
    Args:
        model: trained CVAE model
        cdf: CDF2D instance
        device: cuda or cpu
        n_test_points: number of test points
        n_samples: number of samples per test point
    """
    model.eval()
    
    # Sample test points in workspace
    x_test = []
    for _ in range(n_test_points):
        x = torch.rand(2).to(device)
        x[0] = x[0] * (cdf.task_space[1][0] - cdf.task_space[0][0]) + cdf.task_space[0][0]
        x[1] = x[1] * (cdf.task_space[1][1] - cdf.task_space[0][1]) + cdf.task_space[0][1]
        x_test.append(x)
    
    errors = []
    
    with torch.no_grad():
        for x_idx in x_test:
            # Sample q_0
            q_0 = torch.rand(cdf.num_joints).to(device) * (cdf.q_max - cdf.q_min) + cdf.q_min
            
            # Construct condition
            condition = torch.cat([q_0, x_idx], dim=0)
            
            # Sample from model
            delta_q_samples = model.sample(condition, n_samples=n_samples)
            q_contact_samples = q_0.unsqueeze(0) + delta_q_samples
            
            # Compute forward kinematics for sampled configs
            # Use robot to compute end-effector positions
            obj = Circle(center=x_idx, radius=0.001, device=device)
            gt_cdf = cdf.calculate_cdf(q_0.unsqueeze(0),[obj],method='online_computation',return_grad = False)
            if gt_cdf == float('inf'):
                print('No valid contact configuration for this test point, skipping...')
                continue
            
            # Compute error (should be close to 0 for contact)
            mean_error = torch.abs(gt_cdf).mean().item()
            errors.append(mean_error)
    
    print(f"\nEvaluation Results:")
    print(f"  Mean CDF error: {np.mean(errors):.4f} ± {np.std(errors):.4f}")
    print(f"  Min error: {np.min(errors):.4f}")
    print(f"  Max error: {np.max(errors):.4f}")
    
    return errors

def mean_distance(q_contact_samples):
    """
    Evaluate distances of sampled contact configurations to the target object.
    
    Args:
        q_contact_samples: (n_samples, num_joints) sampled contact configurations
    
    Returns:
        distances(tuple): mean distance and variance to target object
    """
    var_dist = torch.var(torch.norm(q_contact_samples, dim=-1)).item()
    mean_contact = torch.mean(q_contact_samples, dim=0)
    mean_dist = torch.norm(mean_contact).item()
    return mean_dist, var_dist, mean_contact

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize CDF (for evaluation)
    cdf = CDF2D(device)
    
    # Model parameters
    # data_dim = args.num_joints  # Δq dimension
    data_dim = 1  # distance feature dimension
    condition_dim = args.num_joints + 2  # [q_0, x_idx]
    # # Initialize model
    # model = CVAE(
    #     data_dim=data_dim,
    #     condition_dim=condition_dim,
    #     latent_dim=args.latent_dim,
    #     hidden_dims=args.hidden_dims
    # ).to(device)
    # Create model
    model = TCVAE(
        data_dim=data_dim,
        condition_dim=condition_dim,
        latent_dim=args.latent_dim,
        embed_dim=128,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        ff_dim=512,
        dropout=0.1
    ).to(device)
    
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
    # train or test
    print(f'args: {args}')
    if args.train:
        # Get dataloader
        train_dataloader = get_dataloader(
            data_path=args.data_path,
            batch_size=args.batch_size,
            num_joints=args.num_joints,
            samples_per_grid=args.samples_per_grid,
            shuffle=True,
            sampling_strategy='A-q',
            gamma=args.gamma
        )
        
        
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # Training loop
        best_loss = float('inf')
        train_losses = []
        
        print(f"\nStarting training for {args.epochs} epochs...")
        for epoch in range(1, args.epochs + 1):
            # Train
            train_loss, train_recon, train_kl = train_cvae(
                model, train_dataloader, optimizer, device, epoch, beta=args.beta
            )
            
            train_losses.append(train_loss)
            
            print(f"Epoch {epoch}/{args.epochs} - "
                f"Loss: {train_loss:.4f}, Recon: {train_recon:.4f}, KL: {train_kl:.4f}")
            
            # Update learning rate
            scheduler.step(train_loss)
            
            # Save best model
            if train_loss < best_loss:
                best_loss = train_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                }, args.save_path)
                print(f"  -> Saved best model (loss: {train_loss:.4f})")
            
            # # Periodic visualization
            # if epoch % args.vis_interval == 0:
            #     x_test = torch.tensor([0.0, 2.0]).to(device)  # Test point
            #     vis_path = args.save_path.replace('.pth', f'_epoch{epoch}_vis.png')
            #     visualize_samples(model, cdf, x_test, device, n_samples=50, save_path=vis_path)
        
        # Plot training curve
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('CVAE Training Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(args.save_path.replace('.pth', '_training_curve.png'), dpi=150)
        print(f"\nSaved training curve")
    
    # Final evaluation
    print("\n" + "="*50)
    print("Final Evaluation")
    print("="*50)
    
    # Load best model
    checkpoint = torch.load(args.save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate accuracy
    evaluate_accuracy(model, cdf, device, n_test_points=20, n_samples=50)
    
    # Visualize for multiple test points
    test_points = [
        torch.tensor([0.0, 2.0]).to(device),
        torch.tensor([1.0, 1.0]).to(device),
        torch.tensor([-1.0, -1.0]).to(device),
    ]
    
    for i, x_test in enumerate(test_points):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        # vis_path = args.save_path.replace('.pth', f'_final_test{i+1}.png')
        # visualize_samples(model, cdf, x_test, device, n_samples=100, save_path=vis_path,ax = ax)
        # vector_field_vis_path = args.save_path.replace('.pth', f'_vector_field_test{i+1}.png')
        # visualize_vector_field(model, cdf, x_test, device, n_samples=50, save_path=vector_field_vis_path,ax = ax)
        # gradient_magnitude_vis_path = args.save_path.replace('.pth', f'_gradient_magnitude_test{i+1}.png')
        # visualize_gradient_magnitude(model, cdf, x_test, device, n_samples=50, save_path=gradient_magnitude_vis_path,ax = ax)
        d_pred_vis_path = args.save_path.replace('.pth', f'_d_pred_test{i+1}.png')
        visualize_d_pred(model, cdf, x_test, device, n_samples=50, save_path=d_pred_vis_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train CVAE for contact configuration distribution')
    
    # train or test
    parser.add_argument('--train', action='store_true',
                        help='Flag to indicate training mode')
    # Data parameters
    parser.add_argument('--data_path', type=str, default='../2Dexamples/data22.pt',
                        help='Path to .pt data file')
    parser.add_argument('--num_joints', type=int, default=2,
                        help='Number of joints (configuration space dimension)')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='Scaling factor for distance-scaled sampling')
    parser.add_argument('--samples_per_grid', type=int, default=10,
                        help='Number of training samples per grid point')
    
    # Model parameters
    parser.add_argument('--latent_dim', type=int, default=32,
                        help='Dimension of latent space')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 128],
                        help='Hidden layer dimensions')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Weight for KL divergence term')
    
    # Logging parameters
    parser.add_argument('--save_path', type=str, default='./checkpoints/cvae_model.pth',
                        help='Path to save model')
    parser.add_argument('--vis_interval', type=int, default=20,
                        help='Visualization interval (epochs)')
    
    args = parser.parse_args()
    
    # Create checkpoint directory
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    main(args)
