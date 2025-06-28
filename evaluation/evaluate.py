import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

                              
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.fno import FNO2d, divergence
from training.dataset import get_dataloader


def evaluate_model(model, test_loader, device, output_dir, use_amp=False):
    model.eval()
    
                              
    all_rel_l2_errors = []
    all_max_errors = []
    all_div_errors = []
    
                             
    os.makedirs(output_dir, exist_ok=True)
    
                                      
    num_samples = len(test_loader.dataset)
    viz_indices = np.linspace(0, num_samples-1, 5, dtype=int)
    viz_counter = 0
    
                          
    start_time = time.time()
    inference_times = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
                                                      
            inputs = batch['inputs'].to(device)
            targets = batch['targets'].to(device)
            mask = batch['mask'].to(device)
            
                            
            inference_start = time.time()
            
                                                        
            if use_amp and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
            else:
                outputs = model(inputs)
            
                                   
            inference_time = time.time() - inference_start
            inference_times.append(inference_time)
            
                                                      
            for i in range(inputs.shape[0]):
                                       
                output = outputs[i]
                target = targets[i]
                sample_mask = mask[i]
                
                                                   
                output_masked = output * sample_mask
                target_masked = target * sample_mask
                
                                             
                error = output_masked - target_masked
                rel_l2 = torch.norm(error.flatten()) / torch.norm(target_masked.flatten())
                
                                                   
                max_error = torch.max(torch.abs(error))
                
                                            
                div = divergence(output_masked[:2].unsqueeze(0), sample_mask.unsqueeze(0))
                div_error = torch.mean(div**2)
                
                               
                all_rel_l2_errors.append(rel_l2.item())
                all_max_errors.append(max_error.item())
                all_div_errors.append(div_error.item())
                
                                                    
                global_idx = batch_idx * test_loader.batch_size + i
                if global_idx in viz_indices:
                                                           
                    visualize_prediction(
                        output.cpu().numpy(),
                        target.cpu().numpy(),
                        sample_mask.cpu().numpy(),
                        os.path.join(output_dir, f'sample_{viz_counter}_re{inputs[i, 0, 0, 0].item():.2f}.png')
                    )
                    viz_counter += 1
            
                                          
            if torch.cuda.is_available() and batch_idx % 5 == 0:
                torch.cuda.empty_cache()
    
                                    
    total_time = time.time() - start_time
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    
                                                   
    metrics = {
        'mean_rel_l2_error': np.mean(all_rel_l2_errors),
        'std_rel_l2_error': np.std(all_rel_l2_errors),
        'median_rel_l2_error': np.median(all_rel_l2_errors),
        'mean_max_error': np.mean(all_max_errors),
        'mean_div_error': np.mean(all_div_errors),
        'total_evaluation_time': total_time,
        'avg_inference_time_per_batch': avg_inference_time,
        'avg_inference_time_per_sample': avg_inference_time / test_loader.batch_size if test_loader.batch_size > 0 else 0
    }
    
                          
    plot_error_histogram(all_rel_l2_errors, os.path.join(output_dir, 'rel_l2_histogram.png'))
    
                          
    np.save(os.path.join(output_dir, 'metrics.npy'), metrics)
    
                   
    print("\nEvaluation Results:")
    print(f"Mean Relative L2 Error: {metrics['mean_rel_l2_error']:.4f} ± {metrics['std_rel_l2_error']:.4f}")
    print(f"Median Relative L2 Error: {metrics['median_rel_l2_error']:.4f}")
    print(f"Mean Maximum Pointwise Error: {metrics['mean_max_error']:.4f}")
    print(f"Mean Divergence Error: {metrics['mean_div_error']:.6f}")
    print("\nPerformance Metrics:")
    print(f"Total evaluation time: {total_time:.2f} seconds")
    print(f"Average inference time per batch: {avg_inference_time*1000:.2f} ms")
    print(f"Average inference time per sample: {metrics['avg_inference_time_per_sample']*1000:.2f} ms")
    
                                          
    if torch.cuda.is_available():
        print(f"\nGPU Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
        print(f"Cached: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")
    
    return metrics


def visualize_prediction(output, target, mask, save_path):
                    
    u_pred, v_pred, p_pred = output
    u_true, v_true, p_true = target
    
                                  
    vel_mag_pred = np.sqrt(u_pred**2 + v_pred**2)
    vel_mag_true = np.sqrt(u_true**2 + v_true**2)
    
                      
    u_error = np.abs(u_pred - u_true)
    v_error = np.abs(v_pred - v_true)
    p_error = np.abs(p_pred - p_true)
    vel_error = np.abs(vel_mag_pred - vel_mag_true)
    
                   
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
                                   
    for ax in axes.flat:
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
    
                             
    vmax = max(np.max(vel_mag_pred), np.max(vel_mag_true))
    im1 = axes[0, 0].imshow(vel_mag_true, cmap='viridis', origin='lower', vmax=vmax)
    axes[0, 0].set_title('True Velocity Magnitude')
    fig.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(vel_mag_pred, cmap='viridis', origin='lower', vmax=vmax)
    axes[0, 1].set_title('Predicted Velocity Magnitude')
    fig.colorbar(im2, ax=axes[0, 1])
    
    im3 = axes[0, 2].imshow(vel_error, cmap='hot', origin='lower')
    axes[0, 2].set_title('Velocity Magnitude Error')
    fig.colorbar(im3, ax=axes[0, 2])
    
                         
    vmax = max(np.max(np.abs(p_true)), np.max(np.abs(p_pred)))
    im4 = axes[1, 0].imshow(p_true, cmap='coolwarm', origin='lower', vmin=-vmax, vmax=vmax)
    axes[1, 0].set_title('True Pressure')
    fig.colorbar(im4, ax=axes[1, 0])
    
    im5 = axes[1, 1].imshow(p_pred, cmap='coolwarm', origin='lower', vmin=-vmax, vmax=vmax)
    axes[1, 1].set_title('Predicted Pressure')
    fig.colorbar(im5, ax=axes[1, 1])
    
    im6 = axes[1, 2].imshow(p_error, cmap='hot', origin='lower')
    axes[1, 2].set_title('Pressure Error')
    fig.colorbar(im6, ax=axes[1, 2])
    
                           
    vmax = max(np.max(np.abs(u_true)), np.max(np.abs(u_pred)))
    im7 = axes[2, 0].imshow(u_true, cmap='RdBu_r', origin='lower', vmin=-vmax, vmax=vmax)
    axes[2, 0].set_title('True U-Velocity')
    fig.colorbar(im7, ax=axes[2, 0])
    
    im8 = axes[2, 1].imshow(u_pred, cmap='RdBu_r', origin='lower', vmin=-vmax, vmax=vmax)
    axes[2, 1].set_title('Predicted U-Velocity')
    fig.colorbar(im8, ax=axes[2, 1])
    
    im9 = axes[2, 2].imshow(u_error, cmap='hot', origin='lower')
    axes[2, 2].set_title('U-Velocity Error')
    fig.colorbar(im9, ax=axes[2, 2])
    
                                      
    for ax in axes.flat:
        ax.contour(mask, levels=[0.5], colors='k', linewidths=0.5)
    
                 
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_error_histogram(errors, save_path):
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, alpha=0.7, color='blue')
    plt.xlabel('Relative L2 Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Relative L2 Errors')
    plt.axvline(np.mean(errors), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(errors):.4f}')
    plt.axvline(np.median(errors), color='green', linestyle='dashed', linewidth=2, label=f'Median: {np.median(errors):.4f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate FNO model for airfoil flow prediction with GPU acceleration')
    
                     
    parser.add_argument('--data_dir', type=str, default='../data', help='Data directory')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='../evaluation_results', help='Directory to save evaluation results')
    
                                                 
    parser.add_argument('--width', type=int, default=32, help='Width of FNO model')
    parser.add_argument('--modes', type=int, default=12, help='Number of Fourier modes to keep')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of FNO layers')
    
                           
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision for faster evaluation')
    parser.add_argument('--profile', action='store_true', help='Run performance profiling')
    
                       
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID, -1 for CPU')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    
    args = parser.parse_args()
    
                                       
    if torch.cuda.is_available():
        if args.gpu >= 0:
            device = torch.device(f'cuda:{args.gpu}')
        else:
            device = torch.device('cuda')  
        print(f"Using CUDA device: {torch.cuda.get_device_name(device)}")
        print(f"CUDA capability: {torch.cuda.get_device_capability(device)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
        
                                            
        torch.backends.cudnn.benchmark = True                                  
    else:
        device = torch.device('cpu')
        print("CUDA not available. Using CPU.")
    
                                          
    test_loader = get_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
        split='test',
        normalize=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()                                      
    )
    
    print(f"Test dataset size: {len(test_loader.dataset)}")
    
                                                       
    model = FNO2d(
        in_channels=2,                            
        out_channels=3,                      
        width=args.width,
        modes1=args.modes,
        modes2=args.modes,
        n_layers=args.n_layers,
        device=device                        
    )
    
                                
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {args.model_path} (Epoch {checkpoint['epoch']})")
    
                      
    param_count = model.count_params()
    print(f"Model parameters: {param_count:,}")
    
                                            
    if args.profile and torch.cuda.is_available():
        print("\nRunning performance profiling...")
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                                                  
                inputs = next(iter(test_loader))['inputs']
                with torch.no_grad():
                    for _ in range(10):           
                        _ = model(inputs)
                    
                                            
                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    
                    start.record()
                    for _ in range(100):
                        _ = model(inputs)
                    end.record()
                    
                    torch.cuda.synchronize()
                    print(f"Average inference time for batch size {args.batch_size}: {start.elapsed_time(end)/100:.2f} ms")
    
                    
    metrics = evaluate_model(model, test_loader, device, args.output_dir, use_amp=args.mixed_precision)
