import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import sys
from pathlib import Path

                                                        
sys.path.append(str(Path(__file__).parent.parent))

from model.fno import FNO2d
from training.dataset import AirfoilFlowDataset
from torch.utils.data import DataLoader

class Inference:    
    def __init__(self, 
                 model_path, 
                 data_dir='./data',
                 device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
                               
        self.checkpoint = torch.load(model_path, map_location=self.device)
        self.model_args = self.checkpoint['args']
        
                                                                          
        self.model = FNO2d(
            modes1=self.model_args.modes,
            modes2=self.model_args.modes,
            width=self.model_args.width,
            n_layers=self.model_args.n_layers,
            in_channels=2,             
            out_channels=3,           
            use_batch_norm=getattr(self.model_args, 'use_batch_norm', False),
            dropout_rate=getattr(self.model_args, 'dropout_rate', 0.0)
        )
        
                            
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
                            
        self.data_dir = data_dir
        
                              
        self.inference_times = []
        
    def predict_single(self, inputs, enable_dropout=False):                               
        inputs = inputs.to(self.device)
        
                          
        if enable_dropout:
            self.model.train()                                       
        else:
            self.model.eval()                                        
            
                                
        start_time = time.time()
        
                       
        with torch.no_grad():
            outputs = self.model(inputs)
            
                               
        self.inference_times.append(time.time() - start_time)
        
        return outputs
    
    def predict_batch(self, dataloader, return_targets=True, enable_dropout=False):
        predictions = []
        targets = []
        masks = []
        
                          
        if enable_dropout:
            self.model.train()                                       
        else:
            self.model.eval()                                        
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Running inference"):
                                     
                inputs = batch['inputs'].to(self.device)
                mask = batch['mask'].to(self.device)
                
                                        
                start_time = time.time()
                
                               
                outputs = self.model(inputs)
                
                                       
                self.inference_times.append(time.time() - start_time)
                
                               
                predictions.append(outputs.cpu())
                masks.append(mask.cpu())
                
                if return_targets:
                    targets.append(batch['targets'].cpu())
        
        if return_targets:
            return predictions, targets, masks
        else:
            return predictions, masks
    
    def uncertainty_quantification(self, inputs, n_samples=50):
        samples = []
        
                        
        self.model.train()
        
                                       
        with torch.no_grad():
            for _ in tqdm(range(n_samples), desc="MC Sampling"):
                outputs = self.predict_single(inputs, enable_dropout=True)
                samples.append(outputs.cpu())
        
                       
        samples = torch.stack(samples, dim=0)                                                    
        
                                               
        mean = torch.mean(samples, dim=0)
        std = torch.std(samples, dim=0)
        
        return mean, std
    
    def performance_metrics(self, predictions, targets, masks):
        predictions = torch.cat(predictions, dim=0)
        targets = torch.cat(targets, dim=0)
        masks = torch.cat(masks, dim=0)
        
                  
        mse = torch.mean((predictions - targets)**2, dim=(0, 2, 3))
        
                                         
        rel_l2 = []
        for i in range(3):           
            pred_channel = predictions[:, i] * masks
            target_channel = targets[:, i] * masks
            
            error = torch.sqrt(torch.sum((pred_channel - target_channel)**2)) / torch.sqrt(torch.sum(target_channel**2))
            rel_l2.append(error.item())
        
                         
        avg_inference_time = np.mean(self.inference_times)
        
        metrics = {
            'mse': {
                'u': mse[0].item(),
                'v': mse[1].item(),
                'p': mse[2].item(),
                'total': torch.mean(mse).item()
            },
            'rel_l2': {
                'u': rel_l2[0],
                'v': rel_l2[1],
                'p': rel_l2[2],
                'total': np.mean(rel_l2)
            },
            'inference_time': {
                'mean': avg_inference_time,
                'fps': 1.0 / avg_inference_time
            }
        }
        
        return metrics
    
    def run_test_inference(self, batch_size=16, split='test'):
        dataset = AirfoilFlowDataset(self.data_dir, split=split, normalize=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
                       
        predictions, targets, masks = self.predict_batch(dataloader)
        
                           
        metrics = self.performance_metrics(predictions, targets, masks)
        
                       
        print(f"\nPerformance Metrics on {split} set:")
        print(f"MSE Loss: {metrics['mse']['total']:.6f} (u: {metrics['mse']['u']:.6f}, v: {metrics['mse']['v']:.6f}, p: {metrics['mse']['p']:.6f})")
        print(f"Relative L2 Error: {metrics['rel_l2']['total']:.6f} (u: {metrics['rel_l2']['u']:.6f}, v: {metrics['rel_l2']['v']:.6f}, p: {metrics['rel_l2']['p']:.6f})")
        print(f"Inference Time: {metrics['inference_time']['mean']*1000:.2f} ms per batch (FPS: {metrics['inference_time']['fps']:.2f})")
        
        return metrics, predictions, targets, masks

                
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FNO Model Inference")
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], help='Dataset split')
    
    args = parser.parse_args()
    
                                 
    inference = Inference(args.model, args.data_dir)
    
                   
    metrics, _, _, _ = inference.run_test_inference(batch_size=args.batch_size, split=args.split)
