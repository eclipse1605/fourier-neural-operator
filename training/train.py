import os
import time
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

                                                          
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.fno import FNO2d, divergence
from training.dataset import get_dataloader


def train_epoch(model, train_loader, optimizer, device, args, scaler=None):
    """
    Enhanced training function with support for:
    - Mixed precision training
    - Multi-resolution training
    - Higher-order physics constraints
    - Improved logging
    
    Parameters:
    - model: The enhanced FNO model
    - train_loader: DataLoader for training data
    - optimizer: Optimizer
    - device: Device to use (cuda/cpu)
    - args: Command line arguments containing training parameters
    - scaler: GradScaler for mixed precision training
    
    Returns:
    - avg_loss: Average loss over the epoch
    - metrics: Dictionary with detailed training metrics
    """
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_physics = 0.0
    count = 0
    
                               
    pbar = tqdm(train_loader, desc="Training")
    
                           
    batch_times = []
    start_time = time.time()
    
                                              
    scales = [1.0]                                 
    if args.multi_resolution:
        scales = [float(s) for s in args.resolution_scales.split(',')]
    
    for batch_idx, batch in enumerate(pbar):
        batch_start = time.time()
        
                              
                                                
        inputs = batch['inputs'].to(device)
        targets = batch['targets'].to(device)
        mask = batch['mask'].to(device)
        
                                              
        optimizer.zero_grad(set_to_none=True)
        
                                                                                        
        current_scale = 1.0
        if args.multi_resolution and len(scales) > 1 and np.random.random() < 0.5:
            current_scale = np.random.choice(scales)
            
            if current_scale != 1.0:
                                                             
                orig_shape = inputs.shape
                new_h = int(orig_shape[2] * current_scale)
                new_w = int(orig_shape[3] * current_scale)
                
                inputs = F.interpolate(inputs, size=(new_h, new_w), mode='bilinear', align_corners=False)
                targets = F.interpolate(targets, size=(new_h, new_w), mode='bilinear', align_corners=False)
                mask = F.interpolate(mask.unsqueeze(1), size=(new_h, new_w), mode='nearest').squeeze(1)
        
                                               
        if args.mixed_precision and scaler is not None:
            with torch.cuda.amp.autocast():
                                            
                outputs = model(inputs)
                
                                         
                loss_mse = F.mse_loss(outputs, targets)
                
                                                                                         
                div_u = divergence(outputs[:, :2], mask, higher_order=args.higher_order_physics)
                loss_physics = torch.mean(div_u**2)
                
                                               
                loss = loss_mse + args.lambda_physics * loss_physics
            
                                              
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
                                                          
            outputs = model(inputs)
            
                                                 
            loss_mse = F.mse_loss(outputs, targets)
            
                                                                                 
            div_u = divergence(outputs[:, :2], mask, higher_order=args.higher_order_physics)
            loss_physics = torch.mean(div_u**2)
            
                                       
            loss = loss_mse + args.lambda_physics * loss_physics
            
                                                
            loss.backward()
            optimizer.step()
        
                        
        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_mse += loss_mse.item() * batch_size
        total_physics += loss_physics.item() * batch_size
        count += batch_size
        
                              
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        
                             
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'mse': f"{loss_mse.item():.4f}",
            'div': f"{loss_physics.item():.6f}",
            'batch_time': f"{batch_time:.3f}s"
        })
        
                                                           
        if batch_idx % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    epoch_time = time.time() - start_time
    avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
    avg_loss = total_loss / count
    
    print(f"Epoch complete - Avg batch: {avg_batch_time:.3f}s, Total: {epoch_time:.2f}s")
    
    return avg_loss


def validate(model, val_loader, device, args):
    """
    Enhanced validation function with improved metrics and support for uncertainty quantification.
    
    Parameters:
    - model: The enhanced FNO model
    - val_loader: DataLoader for validation data
    - device: Device to use (cuda/cpu)
    - args: Command line arguments with model parameters
    
    Returns:
    - metrics: Dictionary with detailed metrics including uncertainty if enabled
    """
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_physics = 0.0
    rel_l2_error = 0.0
    max_error = 0.0
    count = 0
    
                                    
    uncertainty_metrics = {
        'mean_std': 0.0,                                                 
        'max_std': 0.0,                               
    } if args.dropout_rate > 0 else None
    
                          
    phys_metrics = {
        'div_norm': 0.0,                         
        'max_div': 0.0,                             
    }
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
                                      
                                                    
            inputs = batch['inputs'].to(device)
            targets = batch['targets'].to(device)
            mask = batch['mask'].to(device)
            
                                                           
            if args.mixed_precision and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                                                                               
                    if args.dropout_rate > 0:
                        outputs, std = model(inputs, return_uncertainty=True, n_samples=5)
                                                   
                        uncertainty_metrics['mean_std'] += torch.mean(std).item() * inputs.size(0)
                        uncertainty_metrics['max_std'] += torch.max(std).item() * inputs.size(0)
                    else:
                        outputs = model(inputs)
                    
                                                           
                    loss_mse = F.mse_loss(outputs, targets)
                    
                                                                     
                    div_u = divergence(outputs[:, :2], mask, higher_order=args.higher_order_physics)
                    loss_physics = torch.mean(div_u**2)
                    
                                            
                    phys_metrics['div_norm'] += torch.norm(div_u).item() * inputs.size(0)
                    phys_metrics['max_div'] += torch.max(torch.abs(div_u)).item() * inputs.size(0)
                    
                                                       
                    loss = loss_mse + args.lambda_physics * loss_physics
            else:
                                                         
                if args.dropout_rate > 0:
                    outputs, std = model(inputs, return_uncertainty=True, n_samples=5)
                                               
                    uncertainty_metrics['mean_std'] += torch.mean(std).item() * inputs.size(0)
                    uncertainty_metrics['max_std'] += torch.max(std).item() * inputs.size(0)
                else:
                    outputs = model(inputs)
                
                                                   
                loss_mse = F.mse_loss(outputs, targets)
                
                                                             
                div_u = divergence(outputs[:, :2], mask, higher_order=args.higher_order_physics)
                loss_physics = torch.mean(div_u**2)
                
                                        
                phys_metrics['div_norm'] += torch.norm(div_u).item() * inputs.size(0)
                phys_metrics['max_div'] += torch.max(torch.abs(div_u)).item() * inputs.size(0)
                
                                               
                loss = loss_mse + args.lambda_physics * loss_physics
            
                                        
                                  
            error = outputs - targets
            rel_l2 = torch.norm(error.reshape(error.shape[0], -1), dim=1) / torch.norm(targets.reshape(targets.shape[0], -1), dim=1)
            rel_l2 = torch.mean(rel_l2)
            
                                        
            max_err = torch.max(torch.abs(error))
            
                            
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_mse += loss_mse.item() * batch_size
            total_physics += loss_physics.item() * batch_size
            rel_l2_error += rel_l2.item() * batch_size
            max_error = max(max_error, max_err.item())
            count += batch_size
    
                      
    metrics = {
        'loss': total_loss / count,
        'mse': total_mse / count,
        'physics_loss': total_physics / count,
        'rel_l2_error': rel_l2_error / count,
        'max_error': max_error
    }
    
    model.to(device)
    
                                            
    if args.analyze_modes:
        print("\nAnalyzing important Fourier modes...")
                                                            
        val_batch = next(iter(val_loader))
        inputs = val_batch['inputs'].to(device)
        
                                  
        for layer_idx in range(min(3, args.n_layers)):
            mode_analysis = model.analyze_modes(inputs, layer_idx=layer_idx, top_k=5)
            print(f"\nLayer {layer_idx+1} Top Modes:")
            for i, (mode_i, mode_j) in enumerate(mode_analysis['top_k_modes']):
                print(f"  Mode ({mode_i}, {mode_j}): {mode_analysis['top_k_values'][i]:.4f}")
    
                                                      
    if args.test_resolution_invariance:
        print("\nTesting resolution invariance property...")
        val_batch = next(iter(val_loader))
        inputs = val_batch['inputs'].to(device)
        
                                                
        scale_factors = [0.5, 0.75, 1.25, 1.5, 2.0]
        invariance_results = model.test_resolution_invariance(inputs, scale_factors)
        
        print("Resolution invariance test results:")
        print("Scale Factor | Relative L2 Error")
        print("---------------------------")
        for factor, error in invariance_results.items():
            print(f"{factor:.2f}x       | {error:.6f}")
    
    return metrics


def plot_losses(train_losses, val_losses, save_path):
    """
    Plot training and validation losses.
    
    Parameters:
    - train_losses: List of training losses
    - val_losses: List of validation losses
    - save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, 'b-', label='Training Loss')
    plt.plot(val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def save_model(model, optimizer, scheduler, epoch, metrics, model_path):
    """
    Save the model checkpoint.
    
    Parameters:
    - model: The FNO model
    - optimizer: Optimizer
    - scheduler: Learning rate scheduler
    - epoch: Current epoch
    - metrics: Validation metrics
    - model_path: Path to save the model
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }, model_path)


def train_model(args):
    """
    Enhanced training function with multi-resolution training, improved loss functions,
    and support for the enhanced FNO model features.
    
    Parameters:
    - args: Command-line arguments
    
    Returns:
    - model: Trained model
    """
                                          
    if torch.cuda.is_available():
        if args.gpu >= 0:
            device = torch.device(f'cuda:{args.gpu}')
        else:
            device = torch.device('cuda')
        
                                              
        print(f"Using CUDA device: {torch.cuda.get_device_name(device)}")
        print(f"CUDA capability: {torch.cuda.get_device_capability(device)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
        
                                                               
        if torch.cuda.get_device_capability(device)[0] >= 8:
            print("TensorFloat32 (TF32) is available on this GPU and will be used for training")
            if hasattr(torch.backends.cuda, 'matmul'):
                if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
                    torch.backends.cuda.matmul.allow_tf32 = True
                    print("Enabled TF32 for matrix multiplications")
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True
                print("Enabled TF32 for cuDNN")
    else:
        device = torch.device('cpu')
        print("CUDA not available. Using CPU.")
    
                                 
    os.makedirs(args.output_dir, exist_ok=True)
    
                                         
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)                 
        torch.backends.cudnn.deterministic = args.deterministic                       
        torch.backends.cudnn.benchmark = not args.deterministic                   
    
                                                       
    train_loader = get_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
        split='train',
        normalize=True,
        augment=args.use_augmentation,                            
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.num_workers > 0)
    )
    
    val_loader = get_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
        split='val',
        normalize=True,
        augment=False,                                  
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.num_workers > 0)
    )
    
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(val_loader.dataset)}")
    
                                                                        
    model = FNO2d(
        in_channels=2,                                                            
        out_channels=3,                                           
        width=args.width,
        modes1=args.modes,
        modes2=args.modes,
        n_layers=args.n_layers,
        device=device,
        use_bn=args.use_batch_norm,                                          
        dropout_rate=args.dropout_rate                                                          
    )
    
                                   
    if torch.cuda.is_available() and args.mixed_precision:
        print("Using mixed precision training for better performance")
                                          
    
                         
    total_params = model.count_params()
    print(f"Model parameters: {total_params:,}")
    
                      
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
                                    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    
                                   
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
                                                                
    scaler = None
    if torch.cuda.is_available() and args.mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
    
                                          
    if torch.cuda.is_available():
        print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
        print(f"Initial GPU memory reserved: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")
    
                                               
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
                                   
        start_time = time.time()
        
                                                                                               
        train_loss = train_epoch(
            model, 
            train_loader, 
            optimizer, 
            device, 
            args,                                       
            scaler=scaler
        )
        train_losses.append(train_loss)
        
                                                                                
        val_metrics = validate(
            model, 
            val_loader, 
            device, 
            args                                      
        )
        val_losses.append(val_metrics['loss'])
        
                            
        scheduler.step()
        
                       
        time_per_epoch = time.time() - start_time
        print(f"Epoch {epoch+1} - Time: {time_per_epoch:.2f}s, Train Loss: {train_loss:.6f}, "
              f"Val Loss: {val_metrics['loss']:.6f}, Val Rel L2: {val_metrics['rel_l2_error']:.6f}")
        
                              
        if torch.cuda.is_available():
            print(f"GPU memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
            print(f"GPU memory reserved: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")
                                                         
            torch.cuda.empty_cache()
        
                         
        checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth')
        save_model(model, optimizer, scheduler, epoch+1, val_metrics, checkpoint_path)
        
                         
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_path = os.path.join(args.output_dir, 'model_best.pth')
            save_model(model, optimizer, scheduler, epoch+1, val_metrics, best_model_path)
            print(f"New best model saved! Validation loss: {best_val_loss:.6f}")
        
                     
        plot_losses(train_losses, val_losses, os.path.join(args.output_dir, 'loss_curve.png'))
        
                                        
        config = {
            'args': vars(args),
            'epochs_completed': epoch + 1,
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'training_time': time.time() - start_time
        }
        
        with open(os.path.join(args.output_dir, 'training_config.yaml'), 'w') as f:
            yaml.dump(config, f)
    
    print(f"Training complete! Best validation loss: {best_val_loss:.6f}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train enhanced FNO model for airfoil flow")
    
                     
    parser.add_argument('--data_dir', type=str, default='../data', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='../checkpoints', help='Output directory')
    
                         
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--step_size', type=int, default=100, help='Scheduler step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='Scheduler decay rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    
                      
    parser.add_argument('--width', type=int, default=32, help='Model width')
    parser.add_argument('--modes', type=int, default=12, help='Number of Fourier modes')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--use_batch_norm', action='store_true', help='Use batch normalization')
    parser.add_argument('--dropout_rate', type=float, default=0.0, help='Dropout rate for uncertainty quantification')
    
                              
    parser.add_argument('--lambda_physics', type=float, default=0.1, help='Weight for physics-based loss')
    parser.add_argument('--higher_order_physics', action='store_true', help='Use higher-order derivatives for physics constraints')
    
                                            
    parser.add_argument('--use_augmentation', action='store_true', help='Use data augmentation')
    parser.add_argument('--multi_resolution', action='store_true', help='Use multi-resolution training')
    parser.add_argument('--resolution_scales', type=str, default='0.5,0.75,1.0', 
                        help='Comma-separated list of resolution scales for multi-resolution training')
    
                       
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU ID, -1 for auto-select')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--deterministic', action='store_true', help='Enable deterministic mode')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training')
    parser.add_argument('--profile', action='store_true', help='Profile performance during training')
    
                    
    parser.add_argument('--analyze_modes', action='store_true', help='Analyze important Fourier modes after training')
    parser.add_argument('--test_resolution_invariance', action='store_true', 
                        help='Test resolution invariance property after training')
    
    args = parser.parse_args()
    
                    
    if args.profile and torch.cuda.is_available():
                                      
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                model = train_model(args)
    else:
        model = train_model(args)
