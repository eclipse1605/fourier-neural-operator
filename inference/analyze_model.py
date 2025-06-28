import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
from tqdm import tqdm
import json

sys.path.append(str(Path(__file__).parent.parent))
from inference.inference import Inference
from inference.visualization import FlowVisualizer
sys.path.insert(0, str(Path(__file__).parent))

                      
with open(Path(__file__).parent / 'inference.py') as f:
    inference_code = f.read()
exec(inference_code)

with open(Path(__file__).parent / 'visualization.py') as f:
    visualization_code = f.read()
exec(visualization_code)
from model.fno import FNO2d
from training.dataset import AirfoilFlowDataset, get_dataloader

def analyze_performance(inference, dataset_split='test', batch_size=16, plot=True, save_dir='./results'):
    print(f"\n{'='*40}")
    print(f"Analyzing Performance on {dataset_split} Dataset")
    print(f"{'='*40}")
    
                   
    metrics, predictions, targets, masks = inference.run_test_inference(
        batch_size=batch_size, split=dataset_split
    )
    
                                  
    os.makedirs(save_dir, exist_ok=True)
    
                          
    metrics_file = os.path.join(save_dir, f'metrics_{dataset_split}.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Metrics saved to {metrics_file}")
    
                                           
    if plot:
                                  
        plt.figure(figsize=(10, 6))
        channels = ['u', 'v', 'p', 'total']
        values = [metrics['mse'][c] for c in channels]
        
        plt.bar(channels, values, color=['blue', 'green', 'red', 'purple'])
        plt.title(f'MSE Loss by Channel - {dataset_split}')
        plt.ylabel('MSE Loss')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
                     
        plt.savefig(os.path.join(save_dir, f'mse_loss_{dataset_split}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
                                                
        plt.figure(figsize=(10, 6))
        values = [metrics['rel_l2'][c] for c in channels]
        
        plt.bar(channels, values, color=['blue', 'green', 'red', 'purple'])
        plt.title(f'Relative L2 Error by Channel - {dataset_split}')
        plt.ylabel('Relative L2 Error')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
                     
        plt.savefig(os.path.join(save_dir, f'rel_l2_error_{dataset_split}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print("\nPerformance analysis completed.")
    
    return metrics, predictions, targets, masks

def visualize_predictions(predictions, targets, masks, num_samples=5, save_dir='./results/visualizations'):
    print(f"\n{'='*40}")
    print(f"Generating Visualizations for {num_samples} Test Cases")
    print(f"{'='*40}")
    
                           
    visualizer = FlowVisualizer(save_dir=save_dir)
    
                         
    predictions_cat = torch.cat(predictions, dim=0)
    targets_cat = torch.cat(targets, dim=0)
    masks_cat = torch.cat(masks, dim=0)
    
                                       
    total_samples = predictions_cat.shape[0]
    num_samples = min(num_samples, total_samples)
    
                                                   
    sample_indices = np.linspace(0, total_samples-1, num_samples, dtype=int)
    
                           
    for i, idx in enumerate(sample_indices):
        print(f"Visualizing test case {i+1}/{num_samples} (dataset index {idx})")
        
                                        
        case_dir = os.path.join(save_dir, f'case_{idx:04d}')
        os.makedirs(case_dir, exist_ok=True)
        
                             
        visualizer.visualize_test_case(
            predictions_cat[idx:idx+1], 
            targets_cat[idx:idx+1], 
            masks_cat[idx:idx+1],
            index=0,                                                             
            show=False,
            save=True,
            save_dir=case_dir
        )
    
    print(f"\nVisualizations saved to {save_dir}")

def analyze_uncertainty(inference, dataloader, num_samples=5, mc_samples=30, save_dir='./results/uncertainty'):
    print(f"\n{'='*40}")
    print(f"Performing Uncertainty Quantification on {num_samples} Test Cases")
    print(f"{'='*40}")
    
                           
    visualizer = FlowVisualizer(save_dir=save_dir)
    
                                  
    os.makedirs(save_dir, exist_ok=True)
    
                                 
    samples = []
    targets = []
    masks = []
    
    for batch in dataloader:
        samples.extend([batch['inputs'][i:i+1] for i in range(batch['inputs'].shape[0])])
        targets.extend([batch['targets'][i:i+1] for i in range(batch['targets'].shape[0])])
        masks.extend([batch['mask'][i:i+1] for i in range(batch['mask'].shape[0])])
        
        if len(samples) >= num_samples:
            break
    
                                          
    samples = samples[:num_samples]
    targets = targets[:num_samples]
    masks = masks[:num_samples]
    
                         
    for i, (sample, target, mask) in enumerate(zip(samples, targets, masks)):
        print(f"\nAnalyzing uncertainty for test case {i+1}/{num_samples}")
        
                                        
        case_dir = os.path.join(save_dir, f'case_{i:04d}')
        os.makedirs(case_dir, exist_ok=True)
        
                                            
        mean, std = inference.uncertainty_quantification(sample, n_samples=mc_samples)
        
                               
        visualizer.visualize_uncertainty_analysis(
            mean, std, target, mask,
            index=0,                                                                     
            show=False,
            save=True,
            save_dir=case_dir
        )
        
                                          
        std_np = std.numpy()
        mask_np = mask.numpy()[0]                                
        
                                                                     
        channel_names = ['u-velocity', 'v-velocity', 'pressure']
        uncertainty_stats = {}
        
        for j, name in enumerate(channel_names):
            std_channel = std_np[0, j]                                
            masked_std = std_channel[mask_np > 0.5]
            
            uncertainty_stats[name] = {
                'mean': float(np.mean(masked_std)),
                'max': float(np.max(masked_std)),
                'min': float(np.min(masked_std)),
                'median': float(np.median(masked_std))
            }
        
                                 
        stats_file = os.path.join(case_dir, 'uncertainty_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(uncertainty_stats, f, indent=4)
    
    print(f"\nUncertainty analysis completed. Results saved to {save_dir}")

def analyze_physics(inference, dataloader, num_samples=5, save_dir='./results/physics'):
    print(f"\n{'='*40}")
    print(f"Analyzing Physical Correctness for {num_samples} Test Cases")
    print(f"{'='*40}")
    
                           
    visualizer = FlowVisualizer(save_dir=save_dir)
    
                                  
    os.makedirs(save_dir, exist_ok=True)
    
                                 
    samples = []
    targets = []
    masks = []
    
    for batch in dataloader:
        samples.extend([batch['inputs'][i:i+1] for i in range(batch['inputs'].shape[0])])
        targets.extend([batch['targets'][i:i+1] for i in range(batch['targets'].shape[0])])
        masks.extend([batch['mask'][i:i+1] for i in range(batch['mask'].shape[0])])
        
        if len(samples) >= num_samples:
            break
    
                                          
    samples = samples[:num_samples]
    targets = targets[:num_samples]
    masks = masks[:num_samples]
    
                                         
    div_stats = {
        'prediction': {'mean': [], 'max': []},
        'target': {'mean': [], 'max': []}
    }
    
                         
    for i, (sample, target, mask) in enumerate(zip(samples, targets, masks)):
        print(f"Analyzing physics for test case {i+1}/{num_samples}")
        
                                        
        case_dir = os.path.join(save_dir, f'case_{i:04d}')
        os.makedirs(case_dir, exist_ok=True)
        
                       
        prediction = inference.predict_single(sample)
        
                                         
        pred_np = prediction.cpu().numpy()[0]                          
        target_np = target.cpu().numpy()[0]                            
        mask_np = mask.cpu().numpy()[0]                                
        
                                                                  
        _, pred_div = visualizer.plot_physical_correctness(
            pred_np[0], pred_np[1], mask_np,
            show=False, save=True,
            filename=os.path.join(case_dir, 'pred_divergence.png')
        )
        
                                                              
        _, target_div = visualizer.plot_physical_correctness(
            target_np[0], target_np[1], mask_np,
            show=False, save=True,
            filename=os.path.join(case_dir, 'target_divergence.png')
        )
        
                                               
        mask_bool = mask_np > 0.5
        
                                     
        pred_div_masked = np.abs(pred_div[mask_bool])
        div_stats['prediction']['mean'].append(float(np.mean(pred_div_masked)))
        div_stats['prediction']['max'].append(float(np.max(pred_div_masked)))
        
                                 
        target_div_masked = np.abs(target_div[mask_bool])
        div_stats['target']['mean'].append(float(np.mean(target_div_masked)))
        div_stats['target']['max'].append(float(np.max(target_div_masked)))
    
                                  
    overall_stats = {
        'prediction': {
            'mean_divergence': float(np.mean(div_stats['prediction']['mean'])),
            'max_divergence': float(np.max(div_stats['prediction']['max']))
        },
        'target': {
            'mean_divergence': float(np.mean(div_stats['target']['mean'])),
            'max_divergence': float(np.max(div_stats['target']['max']))
        },
        'ratio': {
            'mean_divergence': float(np.mean(div_stats['prediction']['mean']) / np.mean(div_stats['target']['mean']))
        }
    }
    
                             
    stats_file = os.path.join(save_dir, 'divergence_stats.json')
    with open(stats_file, 'w') as f:
        json.dump({
            'samples': div_stats,
            'overall': overall_stats
        }, f, indent=4)
    
                                   
    plt.figure(figsize=(10, 6))
    x = np.arange(len(div_stats['prediction']['mean']))
    width = 0.35
    
    plt.bar(x - width/2, div_stats['prediction']['mean'], width, label='Prediction')
    plt.bar(x + width/2, div_stats['target']['mean'], width, label='Ground Truth')
    
    plt.xlabel('Sample Index')
    plt.ylabel('Mean Absolute Divergence')
    plt.title('Comparison of Flow Divergence')
    plt.xticks(x)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
                 
    plt.savefig(os.path.join(save_dir, 'divergence_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPhysics analysis completed. Results saved to {save_dir}")
    print(f"Overall mean divergence ratio (prediction/ground_truth): {overall_stats['ratio']['mean_divergence']:.4f}")

def test_resolution_invariance(inference, dataloader, scale_factors=[0.5, 0.75, 1.25, 1.5, 2.0], 
                              num_samples=5, save_dir='./results/resolution'):
    print(f"\n{'='*40}")
    print(f"Testing Resolution Invariance with {len(scale_factors)} Scale Factors")
    print(f"{'='*40}")
    
                                  
    os.makedirs(save_dir, exist_ok=True)
    
                                 
    samples = []
    targets = []
    masks = []
    
    for batch in dataloader:
        samples.extend([batch['inputs'][i:i+1] for i in range(batch['inputs'].shape[0])])
        targets.extend([batch['targets'][i:i+1] for i in range(batch['targets'].shape[0])])
        masks.extend([batch['mask'][i:i+1] for i in range(batch['mask'].shape[0])])
        
        if len(samples) >= num_samples:
            break
    
                                          
    samples = samples[:num_samples]
    targets = targets[:num_samples]
    masks = masks[:num_samples]
    
                                 
    results = {sf: {'rel_l2': []} for sf in scale_factors}
    
                         
    for i, (sample, target, mask) in enumerate(zip(samples, targets, masks)):
        print(f"Testing resolution invariance for test case {i+1}/{num_samples}")
        
                                                        
        baseline_pred = inference.predict_single(sample)
        
                                  
        for sf in scale_factors:
                                                         
            if sf == 1.0:
                continue
                
                                               
            h, w = sample.shape[2], sample.shape[3]
            new_h, new_w = int(h * sf), int(w * sf)
            
                                                  
            resized_sample = torch.nn.functional.interpolate(
                sample, size=(new_h, new_w), mode='bilinear', align_corners=False
            )
            
                                              
            resized_pred = inference.predict_single(resized_sample)
            
                                                                          
            resized_pred_orig = torch.nn.functional.interpolate(
                resized_pred, size=(h, w), mode='bilinear', align_corners=False
            )
            
                                                                                  
            error = torch.sqrt(torch.sum((baseline_pred - resized_pred_orig)**2)) / torch.sqrt(torch.sum(baseline_pred**2))
            results[sf]['rel_l2'].append(float(error.item()))
    
                                                            
    for sf in scale_factors:
        if sf != 1.0:                         
            results[sf]['mean_rel_l2'] = float(np.mean(results[sf]['rel_l2']))
    
                          
    results_file = os.path.join(save_dir, 'resolution_invariance.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
                  
    plt.figure(figsize=(10, 6))
    
                                           
    sfs = [sf for sf in scale_factors if sf != 1.0]
    errors = [results[sf]['mean_rel_l2'] for sf in sfs]
    
    plt.plot(sfs, errors, 'o-', linewidth=2)
    plt.xlabel('Resolution Scale Factor')
    plt.ylabel('Mean Relative L2 Error')
    plt.title('Resolution Invariance Test')
    plt.grid(linestyle='--', alpha=0.7)
    
                 
    plt.savefig(os.path.join(save_dir, 'resolution_invariance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nResolution invariance testing completed. Results saved to {save_dir}")
    print("Mean Relative L2 Error by Scale Factor:")
    for sf in sfs:
        print(f"  {sf:.2f}x: {results[sf]['mean_rel_l2']:.6f}")

def main():
    parser = argparse.ArgumentParser(description="Comprehensive FNO Model Analysis")
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to analyze for each test')
    parser.add_argument('--mc_samples', type=int, default=30, help='Number of Monte Carlo samples for uncertainty quantification')
    parser.add_argument('--skip_performance', action='store_true', help='Skip performance analysis')
    parser.add_argument('--skip_visualization', action='store_true', help='Skip visualization')
    parser.add_argument('--skip_uncertainty', action='store_true', help='Skip uncertainty quantification')
    parser.add_argument('--skip_physics', action='store_true', help='Skip physics analysis')
    parser.add_argument('--skip_resolution', action='store_true', help='Skip resolution invariance testing')
    
    args = parser.parse_args()
    
                             
    os.makedirs(args.output_dir, exist_ok=True)
    
                                 
    inference = Inference(args.model, args.data_dir)
    
                                    
    dataset = AirfoilFlowDataset(args.data_dir, split='test', normalize=True)
    dataloader = get_dataloader(args.data_dir, batch_size=args.batch_size, split='test')
    
                             
    if not args.skip_performance:
        metrics, predictions, targets, masks = analyze_performance(
            inference, dataset_split='test', batch_size=args.batch_size,
            plot=True, save_dir=args.output_dir
        )
    else:
        print("\nSkipping performance analysis...")
                                                  
        _, predictions, targets, masks = inference.run_test_inference(
            batch_size=args.batch_size, split='test'
        )
    
                      
    if not args.skip_visualization:
        visualize_predictions(
            predictions, targets, masks, 
            num_samples=args.num_samples,
            save_dir=os.path.join(args.output_dir, 'visualizations')
        )
    else:
        print("\nSkipping visualization...")
    
                                   
    if not args.skip_uncertainty:
        analyze_uncertainty(
            inference, dataloader,
            num_samples=args.num_samples,
            mc_samples=args.mc_samples,
            save_dir=os.path.join(args.output_dir, 'uncertainty')
        )
    else:
        print("\nSkipping uncertainty quantification...")
    
                         
    if not args.skip_physics:
        analyze_physics(
            inference, dataloader,
            num_samples=args.num_samples,
            save_dir=os.path.join(args.output_dir, 'physics')
        )
    else:
        print("\nSkipping physics analysis...")
    
                              
    if not args.skip_resolution:
        test_resolution_invariance(
            inference, dataloader,
            scale_factors=[0.5, 0.75, 1.25, 1.5, 2.0],
            num_samples=args.num_samples,
            save_dir=os.path.join(args.output_dir, 'resolution')
        )
    else:
        print("\nSkipping resolution invariance testing...")
    
    print(f"\n{'='*60}")
    print(f"Analysis completed! Results saved to {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
