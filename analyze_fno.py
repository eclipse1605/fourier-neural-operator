"""
Simplified FNO Model Analysis Script

This script performs a complete analysis of a trained FNO model without relying on package imports.
"""

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

                                                        
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

                                  
from model.fno import FNO2d
from training.dataset import AirfoilFlowDataset, get_dataloader

class Analyzer:
    """
    FNO Model Analyzer for evaluating performance and generating visualizations
    """
    
    def __init__(self, 
                 model_path, 
                 data_dir='./data',
                 output_dir='./results',
                 device=None):
        """
        Initialize the analyzer
        
        Parameters:
        - model_path: Path to saved model checkpoint
        - data_dir: Directory containing the dataset
        - output_dir: Directory to save results
        - device: Device to run analysis on
        """
                    
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
                               
        self.checkpoint = torch.load(model_path, map_location=self.device)
        
                                 
        if 'args' in self.checkpoint:
            self.config = self.checkpoint['args']
            width = self.config.width
            modes = self.config.modes
            n_layers = self.config.n_layers
            use_batch_norm = getattr(self.config, 'use_batch_norm', False)
            dropout_rate = getattr(self.config, 'dropout_rate', 0.0)
        else:
            print("No configuration found in checkpoint, using defaults")
            width = 32
            modes = 12
            n_layers = 4
            use_batch_norm = True                                                            
            dropout_rate = 0.0
        
                          
        self.model = FNO2d(
            modes1=modes,
            modes2=modes,
            width=width,
            n_layers=n_layers,
            in_channels=2,             
            out_channels=3,           
            use_bn=use_batch_norm,
            dropout_rate=dropout_rate
        )
        
                            
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
                            
        self.data_dir = data_dir
        
                              
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def run_analysis(self, batch_size=16, num_samples=5):
        """
        Run a complete analysis of the model
        
        Parameters:
        - batch_size: Batch size for inference
        - num_samples: Number of samples to analyze
        """
        print(f"\n{'='*60}")
        print(f"Running FNO Model Analysis")
        print(f"{'='*60}")
        
                                         
        test_dataset = AirfoilFlowDataset(self.data_dir, split='test', normalize=True)
        test_loader = get_dataloader(self.data_dir, batch_size=batch_size, split='test')
        
        print(f"Test dataset size: {len(test_dataset)}")
        
                           
        metrics = self.evaluate_performance(test_loader)
        
                                 
        self.generate_visualizations(test_loader, num_samples)
        
                      
        metrics_file = os.path.join(self.output_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        print(f"\nAnalysis completed. Results saved to {self.output_dir}")
    
    def evaluate_performance(self, dataloader):
        """
        Evaluate model performance
        
        Parameters:
        - dataloader: DataLoader for test data
        
        Returns:
        - metrics: Dictionary of performance metrics
        """
        print(f"\n{'='*40}")
        print(f"Evaluating Model Performance")
        print(f"{'='*40}")
        
        all_rel_l2_errors = []
        all_div_errors = []
        inference_times = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                                     
                inputs = batch['inputs'].to(self.device)
                targets = batch['targets'].to(self.device)
                mask = batch['mask'].to(self.device)
                
                                        
                start_time = time.time()
                outputs = self.model(inputs)
                torch.cuda.synchronize()
                inference_times.append(time.time() - start_time)
                
                                                   
                for i in range(inputs.shape[0]):
                    output = outputs[i]
                    target = targets[i]
                    sample_mask = mask[i]
                    
                                
                    output_masked = output * sample_mask
                    target_masked = target * sample_mask
                    
                                                 
                    error = output_masked - target_masked
                    rel_l2 = torch.norm(error.flatten()) / torch.norm(target_masked.flatten())
                    all_rel_l2_errors.append(rel_l2.item())
                    
                                                
                    from model.fno import divergence
                    div = divergence(output_masked[:2].unsqueeze(0), sample_mask.unsqueeze(0))
                    div_error = torch.mean(div**2)
                    all_div_errors.append(div_error.item())
        
                           
        metrics = {
            'mean_rel_l2_error': float(np.mean(all_rel_l2_errors)),
            'median_rel_l2_error': float(np.median(all_rel_l2_errors)),
            'std_rel_l2_error': float(np.std(all_rel_l2_errors)),
            'min_rel_l2_error': float(np.min(all_rel_l2_errors)),
            'max_rel_l2_error': float(np.max(all_rel_l2_errors)),
            'mean_div_error': float(np.mean(all_div_errors)),
            'inference_time_ms': float(np.mean(inference_times) * 1000),
            'inference_fps': float(1.0 / np.mean(inference_times))
        }
        
                       
        print("\nPerformance Metrics:")
        print(f"Mean Relative L2 Error: {metrics['mean_rel_l2_error']:.6f} ± {metrics['std_rel_l2_error']:.6f}")
        print(f"Median Relative L2 Error: {metrics['median_rel_l2_error']:.6f}")
        print(f"Mean Divergence Error: {metrics['mean_div_error']:.6f}")
        print(f"Inference Time: {metrics['inference_time_ms']:.2f} ms per batch (FPS: {metrics['inference_fps']:.2f})")
        
        return metrics
    
    def generate_visualizations(self, dataloader, num_samples=5):
        """
        Generate visualizations for model predictions
        
        Parameters:
        - dataloader: DataLoader for test data
        - num_samples: Number of samples to visualize
        """
        print(f"\n{'='*40}")
        print(f"Generating Visualizations")
        print(f"{'='*40}")
        
                                             
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
                                                       
        num_batches = len(dataloader)
        batch_indices = np.linspace(0, num_batches-1, num_samples, dtype=int)
        
                                    
        viz_counter = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating visualizations")):
                if batch_idx not in batch_indices:
                    continue
                    
                                     
                inputs = batch['inputs'].to(self.device)
                targets = batch['targets'].to(self.device)
                mask = batch['mask'].to(self.device)
                
                                      
                outputs = self.model(inputs)
                
                                                                
                self.visualize_prediction(
                    outputs[0].cpu().numpy(),
                    targets[0].cpu().numpy(),
                    mask[0].cpu().numpy(),
                    os.path.join(viz_dir, f'sample_{viz_counter}.png')
                )
                
                viz_counter += 1
                
                if viz_counter >= num_samples:
                    break
    
    def visualize_prediction(self, output, target, mask, save_path):
        """
        Generate visualization comparing predicted and target fields
        
        Parameters:
        - output: Predicted output [channels, height, width]
        - target: Target output [channels, height, width]
        - mask: Binary mask [height, width]
        - save_path: Path to save visualization
        """
                       
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        
                        
        u_pred, v_pred, p_pred = output
        u_true, v_true, p_true = target
        
                                      
        vel_mag_pred = np.sqrt(u_pred**2 + v_pred**2)
        vel_mag_true = np.sqrt(u_true**2 + v_true**2)
        
                          
        u_error = np.abs(u_pred - u_true)
        v_error = np.abs(v_pred - v_true)
        p_error = np.abs(p_pred - p_true)
        vel_error = np.abs(vel_mag_pred - vel_mag_true)
        
                              
        titles = ['True', 'Predicted', 'Error']
        fields = [
            [vel_mag_true, vel_mag_pred, vel_error],
            [u_true, u_pred, u_error],
            [v_true, v_pred, v_error]
        ]
        field_names = ['Velocity Magnitude', 'U-Velocity', 'V-Velocity']
        cmaps = ['viridis', 'RdBu_r', 'RdBu_r']
        
                         
        for i, (field_row, field_name, cmap) in enumerate(zip(fields, field_names, cmaps)):
            for j, (field, title) in enumerate(zip(field_row, titles)):
                            
                field = field * mask
                
                            
                if j == 2:              
                    im = axes[i, j].imshow(field, cmap='hot', origin='lower')
                else:
                    if i > 0:                                   
                        vmax = max(np.max(np.abs(field_row[0])), np.max(np.abs(field_row[1])))
                        im = axes[i, j].imshow(field, cmap=cmap, origin='lower', vmin=-vmax, vmax=vmax)
                    else:                                     
                        vmax = max(np.max(field_row[0]), np.max(field_row[1]))
                        im = axes[i, j].imshow(field, cmap=cmap, origin='lower', vmax=vmax)
                
                              
                plt.colorbar(im, ax=axes[i, j])
                
                           
                axes[i, j].set_title(f"{title} {field_name}")
                
                                     
                axes[i, j].contour(mask, levels=[0.5], colors='k', linewidths=1.0)
        
                       
        plt.tight_layout()
        
                     
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="FNO Model Analysis")
    parser.add_argument('--model', dest='model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to analyze')
    
    args = parser.parse_args()
    
                     
    analyzer = Analyzer(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
                  
    analyzer.run_analysis(
        batch_size=args.batch_size,
        num_samples=args.num_samples
    )

if __name__ == "__main__":
    main()
