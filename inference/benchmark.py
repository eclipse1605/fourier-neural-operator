import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import psutil
import sys
from pathlib import Path
from tqdm import tqdm

                                                        
sys.path.append(str(Path(__file__).parent.parent))

from inference.inference import Inference

def measure_inference_time(model, input_tensor, warmup=10, trials=100, device='cuda'):
    """
    Measure inference time for a model
    
    Parameters:
    - model: PyTorch model
    - input_tensor: Input tensor
    - warmup: Number of warmup runs
    - trials: Number of timed trials
    - device: Device to run inference on
    
    Returns:
    - times: List of inference times in milliseconds
    """
                          
    input_tensor = input_tensor.to(device)
    
                 
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
    
                               
    if device == 'cuda':
        torch.cuda.synchronize()
    
                 
    times = []
    with torch.no_grad():
        for _ in range(trials):
            if device == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            _ = model(input_tensor)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append((end_time - start_time) * 1000)                 
    
    return times

def measure_memory_usage(model, input_tensor, device='cuda'):
    """
    Measure memory usage for a model
    
    Parameters:
    - model: PyTorch model
    - input_tensor: Input tensor
    - device: Device to run inference on
    
    Returns:
    - memory_stats: Dictionary of memory statistics
    """
                            
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
                       
    cpu_memory_before = psutil.Process().memory_info().rss / (1024 * 1024)      
    
                       
    if device == 'cuda':
        gpu_memory_before = torch.cuda.memory_allocated() / (1024 * 1024)      
    else:
        gpu_memory_before = 0
    
                   
    with torch.no_grad():
        _ = model(input_tensor.to(device))
    
                      
    cpu_memory_after = psutil.Process().memory_info().rss / (1024 * 1024)      
    
                      
    if device == 'cuda':
        gpu_memory_after = torch.cuda.memory_allocated() / (1024 * 1024)
        gpu_memory_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        gpu_memory_after = 0
        gpu_memory_peak = 0
    
    memory_stats = {
        'cpu_memory_before_mb': cpu_memory_before,
        'cpu_memory_after_mb': cpu_memory_after,
        'cpu_memory_used_mb': cpu_memory_after - cpu_memory_before,
        'gpu_memory_before_mb': gpu_memory_before,
        'gpu_memory_after_mb': gpu_memory_after,
        'gpu_memory_used_mb': gpu_memory_after - gpu_memory_before,
        'gpu_memory_peak_mb': gpu_memory_peak
    }
    
    return memory_stats

def benchmark_resolution_scaling(inference, base_resolution=(64, 64), 
                               scale_factors=[0.5, 1.0, 1.5, 2.0, 3.0, 4.0],
                               batch_size=1, trials=100):
    """
    Benchmark model performance across different resolutions
    
    Parameters:
    - inference: Inference object
    - base_resolution: Base resolution (height, width)
    - scale_factors: Resolution scaling factors to test
    - batch_size: Batch size for inference
    - trials: Number of trials for timing
    
    Returns:
    - results: Dictionary of benchmarking results
    """
    results = {}
    
                                                           
    channels = 2             
    base_h, base_w = base_resolution
    
                      
    model = inference.model
    device = inference.device
    
                                 
    for sf in scale_factors:
                              
        h, w = int(base_h * sf), int(base_w * sf)
        resolution = (h, w)
        
        print(f"Benchmarking resolution {resolution} (scale factor {sf}x)...")
        
                            
        dummy_input = torch.randn(batch_size, channels, h, w, dtype=torch.float32)
        
                                
        times = measure_inference_time(model, dummy_input, warmup=10, trials=trials, device=device)
        
                              
        memory_stats = measure_memory_usage(model, dummy_input, device=device)
        
                        
        results[sf] = {
            'resolution': resolution,
            'inference_time_ms': {
                'mean': float(np.mean(times)),
                'median': float(np.median(times)),
                'min': float(np.min(times)),
                'max': float(np.max(times)),
                'std': float(np.std(times))
            },
            'memory': memory_stats,
            'throughput_fps': float(1000 / np.mean(times))
        }
    
    return results

def benchmark_batch_size_scaling(inference, resolution=(64, 64), 
                               batch_sizes=[1, 2, 4, 8, 16, 32, 64],
                               trials=100):
    """
    Benchmark model performance across different batch sizes
    
    Parameters:
    - inference: Inference object
    - resolution: Input resolution (height, width)
    - batch_sizes: Batch sizes to test
    - trials: Number of trials for timing
    
    Returns:
    - results: Dictionary of benchmarking results
    """
    results = {}
    
                                                           
    channels = 2             
    h, w = resolution
    
                      
    model = inference.model
    device = inference.device
    
                               
    for bs in batch_sizes:
        print(f"Benchmarking batch size {bs}...")
        
                            
        dummy_input = torch.randn(bs, channels, h, w, dtype=torch.float32)
        
                                                   
        if device == 'cuda':
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1024)      
            estimated_mem = bs * channels * h * w * 4 * 5 / (1024 * 1024 * 1024)                       
            
            if estimated_mem > gpu_mem * 0.9:
                print(f"  Skipping batch size {bs} - estimated memory ({estimated_mem:.2f} GB) exceeds available GPU memory ({gpu_mem:.2f} GB)")
                continue
        
        try:
                                    
            times = measure_inference_time(model, dummy_input, warmup=5, trials=trials, device=device)
            
                                  
            memory_stats = measure_memory_usage(model, dummy_input, device=device)
            
                                                        
            throughput = bs * 1000 / np.mean(times)               
            
                            
            results[bs] = {
                'inference_time_ms': {
                    'mean': float(np.mean(times)),
                    'median': float(np.median(times)),
                    'min': float(np.min(times)),
                    'max': float(np.max(times)),
                    'std': float(np.std(times))
                },
                'memory': memory_stats,
                'throughput_samples_per_sec': float(throughput)
            }
            
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"  Skipping batch size {bs} - out of memory")
                          
                if device == 'cuda':
                    torch.cuda.empty_cache()
            else:
                raise e
    
    return results

def plot_benchmark_results(resolution_results, batch_results, save_dir='./results/benchmarks'):
    """
    Plot benchmarking results
    
    Parameters:
    - resolution_results: Results from resolution scaling benchmark
    - batch_results: Results from batch size scaling benchmark
    - save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
                                                 
    plt.figure(figsize=(10, 6))
    
                  
    scale_factors = sorted(list(resolution_results.keys()))
    times = [resolution_results[sf]['inference_time_ms']['mean'] for sf in scale_factors]
    
                                                             
    base_time = times[scale_factors.index(1.0)]
    base_n = 1.0
    theoretical_times = [base_time * (sf**2) * (np.log(sf**2) / np.log(base_n**2)) for sf in scale_factors]
    
    plt.plot(scale_factors, times, 'o-', linewidth=2, label='Measured')
    plt.plot(scale_factors, theoretical_times, '--', linewidth=2, label='Theoretical O(n log n)')
    
    plt.xlabel('Resolution Scale Factor')
    plt.ylabel('Inference Time (ms)')
    plt.title('FNO Inference Time vs. Resolution')
    plt.grid(linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.savefig(os.path.join(save_dir, 'resolution_time.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
                                               
    plt.figure(figsize=(10, 6))
    
                  
    gpu_memory = [resolution_results[sf]['memory']['gpu_memory_used_mb'] for sf in scale_factors]
    
    plt.plot(scale_factors, gpu_memory, 'o-', linewidth=2)
    plt.xlabel('Resolution Scale Factor')
    plt.ylabel('GPU Memory Usage (MB)')
    plt.title('FNO Memory Usage vs. Resolution')
    plt.grid(linestyle='--', alpha=0.7)
    
    plt.savefig(os.path.join(save_dir, 'resolution_memory.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
                                             
    plt.figure(figsize=(10, 6))
    
                  
    batch_sizes = sorted(list(batch_results.keys()))
    throughput = [batch_results[bs]['throughput_samples_per_sec'] for bs in batch_sizes]
    
    plt.plot(batch_sizes, throughput, 'o-', linewidth=2)
    plt.xlabel('Batch Size')
    plt.ylabel('Throughput (samples/second)')
    plt.title('FNO Throughput vs. Batch Size')
    plt.grid(linestyle='--', alpha=0.7)
    
    plt.savefig(os.path.join(save_dir, 'batch_throughput.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
                                                  
    plt.figure(figsize=(10, 6))
    
                  
    time_per_sample = [batch_results[bs]['inference_time_ms']['mean'] / bs for bs in batch_sizes]
    
    plt.plot(batch_sizes, time_per_sample, 'o-', linewidth=2)
    plt.xlabel('Batch Size')
    plt.ylabel('Time per Sample (ms)')
    plt.title('FNO Processing Time per Sample vs. Batch Size')
    plt.grid(linestyle='--', alpha=0.7)
    
                                                           
    if 1 in batch_results:
        single_sample_time = batch_results[1]['inference_time_ms']['mean']
        plt.axhline(y=single_sample_time, color='r', linestyle='--', label=f'Single Sample ({single_sample_time:.2f} ms)')
        plt.legend()
    
    plt.savefig(os.path.join(save_dir, 'batch_time_per_sample.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to run benchmarks"""
    import argparse
    
    parser = argparse.ArgumentParser(description="FNO Model Performance Benchmarks")
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./results/benchmarks', help='Output directory for results')
    parser.add_argument('--resolution_scaling', action='store_true', help='Run resolution scaling benchmark')
    parser.add_argument('--batch_scaling', action='store_true', help='Run batch size scaling benchmark')
    parser.add_argument('--base_resolution', type=int, nargs=2, default=[64, 64], help='Base resolution (height width)')
    parser.add_argument('--trials', type=int, default=100, help='Number of trials for timing')
    
    args = parser.parse_args()
    
                             
    os.makedirs(args.output_dir, exist_ok=True)
    
                                 
    inference = Inference(args.model)
    
                       
    print(f"\n{'='*60}")
    print(f"System Information:")
    print(f"{'='*60}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA compute capability: {torch.cuda.get_device_capability(0)}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1024)      
        print(f"GPU memory: {total_memory:.2f} GB")
    print(f"CPU: {psutil.cpu_count(logical=False)} physical cores, {psutil.cpu_count(logical=True)} logical cores")
    
                       
    num_params = sum(p.numel() for p in inference.model.parameters())
    print(f"\nModel parameters: {num_params:,}")
    
                                   
    benchmark_results = {
        'system_info': {
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available()
        },
        'model_info': {
            'parameters': num_params
        }
    }
    
    if torch.cuda.is_available():
        benchmark_results['system_info'].update({
            'cuda_device': torch.cuda.get_device_name(0),
            'cuda_capability': torch.cuda.get_device_capability(0),
            'gpu_memory_gb': total_memory
        })
    
    benchmark_results['system_info'].update({
        'cpu_physical_cores': psutil.cpu_count(logical=False),
        'cpu_logical_cores': psutil.cpu_count(logical=True)
    })
    
                    
    if args.resolution_scaling:
        print(f"\n{'='*60}")
        print(f"Resolution Scaling Benchmark")
        print(f"{'='*60}")
        
        resolution_results = benchmark_resolution_scaling(
            inference,
            base_resolution=tuple(args.base_resolution),
            scale_factors=[0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0],
            trials=args.trials
        )
        
        benchmark_results['resolution_scaling'] = resolution_results
        
                                 
        resolution_file = os.path.join(args.output_dir, 'resolution_scaling.json')
        with open(resolution_file, 'w') as f:
            json.dump(resolution_results, f, indent=4)
        
        print(f"Resolution scaling results saved to {resolution_file}")
    
    if args.batch_scaling:
        print(f"\n{'='*60}")
        print(f"Batch Size Scaling Benchmark")
        print(f"{'='*60}")
        
        batch_results = benchmark_batch_size_scaling(
            inference,
            resolution=tuple(args.base_resolution),
            batch_sizes=[1, 2, 4, 8, 16, 32, 64, 128],
            trials=args.trials
        )
        
        benchmark_results['batch_scaling'] = batch_results
        
                            
        batch_file = os.path.join(args.output_dir, 'batch_scaling.json')
        with open(batch_file, 'w') as f:
            json.dump(batch_results, f, indent=4)
        
        print(f"Batch scaling results saved to {batch_file}")
    
                           
    combined_file = os.path.join(args.output_dir, 'benchmark_results.json')
    with open(combined_file, 'w') as f:
        json.dump(benchmark_results, f, indent=4)
    
    print(f"Combined results saved to {combined_file}")
    
                                              
    if args.resolution_scaling and args.batch_scaling:
        plot_benchmark_results(
            resolution_results=benchmark_results['resolution_scaling'],
            batch_results=benchmark_results['batch_scaling'],
            save_dir=args.output_dir
        )
        print(f"Benchmark plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
