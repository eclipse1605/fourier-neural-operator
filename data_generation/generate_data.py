import os
import torch
import argparse
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from airfoil import generate_structured_grid
from navier_stokes import NavierStokesGPUSolver

def generate_case_gpu(grid_size, Re, airfoil_type='0012', n_steps=5000, dt=0.001, max_attempts=3, device=None, batch_id=None):
    batch_str = f"[Batch {batch_id}]" if batch_id is not None else ""
    try:
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        grid = generate_structured_grid(grid_size, airfoil_type)
        mask = grid['mask']
        
        mask_tensor = torch.from_numpy(mask).float().to(device)
        
        for attempt in range(max_attempts):
            try:
                current_dt = dt / (2**attempt)                                      
                current_steps = n_steps * (2**attempt)                                        
                
                torch.cuda.empty_cache()
                
                solver = NavierStokesGPUSolver(
                    grid_size=grid_size, 
                    Re=Re, 
                    dt=current_dt, 
                    airfoil_mask=mask_tensor,
                    device=device
                )
                
                solver.set_boundary_conditions(u_in=1.0)
                
                results = solver.run_simulation(
                    n_steps=current_steps, 
                    check_steady=True, 
                    steady_tol=1e-4, 
                    verbosity=0
                )
                
                if results['failed_steps'] > 10 or np.any(np.isnan(results['u'])) or np.any(np.isnan(results['v'])):
                    if attempt == max_attempts - 1:
                        print(f"{batch_str} Warning: Simulation for Re={Re} may be unstable")
                    else:
                        continue
                
                case_data = {
                    'u': results['u'],
                    'v': results['v'],
                    'p': results['p'],
                    'mask': mask,
                    'Re': Re,
                    'airfoil_type': airfoil_type,
                    'grid_size': grid_size,
                    'steps': results['steps'],
                    'steady': results['steady'],
                    'dt_final': results['dt_final']
                }
                
                solver.to_cpu()
                del solver, mask_tensor
                torch.cuda.empty_cache()
                
                return case_data
            
            except Exception as e:
                print(f"{batch_str} Attempt {attempt+1} failed for Re={Re}: {str(e)}")
                if attempt == max_attempts - 1:
                    print(f"{batch_str} Failed to simulate case with Re={Re} after {max_attempts} attempts")
                    torch.cuda.empty_cache()
                    return None
    
    except Exception as e:
        print(f"{batch_str} Fatal error generating case for Re={Re}: {str(e)}")
        torch.cuda.empty_cache()
        return None
    
    torch.cuda.empty_cache()
    return None                                           


def multi_gpu_case_generation(params_list, max_workers=None):
    params_list = sorted(params_list, key=lambda p: p.get('Re', 0), reverse=True)
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpus == 0:
        print("No GPUs available, running on CPU")
        max_workers = 1
    else:
        if max_workers is None:
            max_workers = n_gpus
        print(f"Using {max_workers} workers across {n_gpus} GPUs")
    
    for i, params in enumerate(params_list):
        params['batch_id'] = i + 1
    
    def worker_function(params):
        """Worker function with GPU assignment"""
        batch_id = params.pop('batch_id', None)
        if n_gpus > 0:
            worker_id = params.get('worker_id', 0) % n_gpus
            gpu_id = worker_id
            device = torch.device(f'cuda:{gpu_id}')
            print(f"[Batch {batch_id}] Assigned to GPU {gpu_id} for Re={params.get('Re', 'unknown')}")
        else:
            device = torch.device('cpu')
            print(f"[Batch {batch_id}] Running on CPU for Re={params.get('Re', 'unknown')}")
        if 'worker_id' in params:
            params.pop('worker_id')
        return generate_case_gpu(**params, device=device, batch_id=batch_id)
    
    for i, params in enumerate(params_list):
        params['worker_id'] = i % max_workers
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker_function, p.copy()) for p in params_list]
        
        progress_bar = tqdm(
            total=len(futures),
            desc="Generating cases",
            position=0,
            leave=True,
            ncols=100,
            bar_format='{l_bar}{bar:30}{r_bar}'
        )
        
        for i, future in enumerate(futures):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                                                      
                progress_bar.update(1)
                                                      
                progress_bar.set_description(f"Generating cases ({len(results)}/{i+1} successful)")
            except Exception as e:
                progress_bar.update(1)
                print(f"\nError in worker: {str(e)}")
        progress_bar.close()
    return results

def distribute_re_progressive(re_min, re_max, n_samples):
    log_min = np.log10(re_min)
    log_max = np.log10(re_max)
    exponent = 0.8                                           
    t = np.linspace(0, 1, n_samples) ** exponent
    log_values = log_min + t * (log_max - log_min)
    re_values = 10 ** log_values
    
                                           
    re_values[0] = re_min
    re_values[-1] = re_max
    
    return re_values


def create_dataset_gpu(output_dir, re_min, re_max, n_samples, grid_size=(128, 128), airfoil_types=None, seed=42, progressive=True, max_workers=None, batch_size=None, resume=True):
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    if airfoil_types is None:
        airfoil_types = ['0012']
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"Found {n_gpus} CUDA devices:")
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"  Device {i}: {props.name}")
            print(f"    Compute Capability: {props.major}.{props.minor}")
            print(f"    Total Memory: {props.total_memory / 1e9:.2f} GB")
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
        if max_workers is None:
            max_workers = n_gpus
        torch.backends.cudnn.benchmark = True
        print(f"Using {max_workers} workers with {n_gpus} GPUs")
    else:
        print("CUDA not available. Using CPU instead.")
        max_workers = 1
    if progressive:
        re_values = distribute_re_progressive(re_min, re_max, n_samples)
    else:
        re_values = np.random.uniform(re_min, re_max, n_samples)
    re_values = np.sort(re_values)[::-1]                                                       
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, "generation_checkpoint.npz")
                                                      
    existing_files = set()
    if resume and os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            if filename.endswith('.npz') and filename.startswith('case_Re'):
                existing_files.add(filename)
        print(f"Found {len(existing_files)} existing case files in the output directory")
                                 
    completed_params = []
    completed_cases = []
    if resume and os.path.exists(checkpoint_file):
        try:
            checkpoint_data = np.load(checkpoint_file, allow_pickle=True)
            completed_params = checkpoint_data['completed_params'].tolist() if 'completed_params' in checkpoint_data else []
                                                                                                  
                                              
            completed_count = checkpoint_data['completed_count'].item() if 'completed_count' in checkpoint_data else 0
            print(f"Resuming from checkpoint: {completed_count} cases already processed")
        except Exception as e:
            print(f"Warning: Could not load checkpoint file: {e}")
            completed_params = []
            completed_cases = []
    
                           
    params_list = []
    for i, re in enumerate(re_values):
                                                                            
        airfoil_type = np.random.choice(airfoil_types)
        
                                     
        params = {
            'grid_size': grid_size,
            'Re': float(re),
            'airfoil_type': airfoil_type,
            'n_steps': 5000,
            'dt': 0.001,
            'max_attempts': 3
        }
        
                                                  
        expected_filename = f"case_Re{params['Re']:.0f}_{params['airfoil_type']}.npz"
        if expected_filename in existing_files or params in completed_params:
            print(f"Skipping already processed case: Re={params['Re']:.0f}, airfoil={params['airfoil_type']}")
            continue
            
        params_list.append(params)
    
                                              
    if not params_list:
        print("All cases have already been processed. Nothing to do.")
        return completed_cases
    
                                                         
    if batch_size is None:
        batch_size = len(params_list)                       
    
                        
    param_batches = [params_list[i:i+batch_size] for i in range(0, len(params_list), batch_size)]
    successful_cases = []
    for batch_idx, batch_params in enumerate(param_batches):
        print(f"Processing batch {batch_idx+1}/{len(param_batches)} with {len(batch_params)} cases")
        batch_results = multi_gpu_case_generation(batch_params, max_workers=max_workers)
        successful_cases.extend(batch_results)
        print(f"Batch {batch_idx+1} complete: {len(batch_results)}/{len(batch_params)} cases successful")
        
                                              
        completed_params.extend(batch_params)
        completed_cases.extend(batch_results)
        
                                          
        try:
            np.savez_compressed(
                checkpoint_file,
                completed_params=np.array(completed_params, dtype=object),
                completed_count=len(completed_cases)
            )
            print(f"Checkpoint saved: {len(completed_cases)} cases processed so far")
        except Exception as e:
            print(f"Warning: Could not save checkpoint: {e}")
        
                                   
        torch.cuda.empty_cache()
    total_success_rate = 100 * len(successful_cases) / len(params_list)
    print(f"Successfully generated {len(successful_cases)} out of {len(params_list)} cases ({total_success_rate:.1f}%)")
    
                                           
    save_progress = tqdm(
        total=len(successful_cases),
        desc="Saving cases", 
        position=0, 
        leave=True,
        ncols=100,
        bar_format='{l_bar}{bar:30}{r_bar}'
    )
    
    for i, case_data in enumerate(successful_cases):
        filename = f"case_Re{case_data['Re']:.0f}_{case_data['airfoil_type']}.npz"
        filepath = os.path.join(output_dir, filename)
        np.savez_compressed(
            filepath,
            u=case_data['u'],
            v=case_data['v'],
            p=case_data['p'],
            mask=case_data['mask'],
            Re=case_data['Re'],
            airfoil_type=case_data['airfoil_type']
        )
        if i % 10 == 0:
            viz_dir = os.path.join(output_dir, 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            viz_filename = os.path.join(viz_dir, f"case_Re{case_data['Re']:.0f}_{case_data['airfoil_type']}.png")
            visualize_case(case_data, viz_filename)
    
    print(f"Dataset generation complete. {len(successful_cases)} cases saved to {output_dir}")
    
    return successful_cases


def visualize_case(case_data, output_file=None):
    u = case_data['u']
    v = case_data['v']
    p = case_data['p']
    mask = case_data['mask']
    Re = case_data['Re']
    velocity_mag = np.sqrt(u**2 + v**2)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    im0 = axs[0, 0].imshow(velocity_mag, cmap='jet')
    axs[0, 0].set_title(f'Velocity Magnitude (Re={Re:.0f})')
    plt.colorbar(im0, ax=axs[0, 0])
    
                   
    im1 = axs[0, 1].imshow(p, cmap='coolwarm')
    axs[0, 1].set_title('Pressure')
    plt.colorbar(im1, ax=axs[0, 1])
    
                      
    y, x = np.mgrid[0:u.shape[0], 0:u.shape[1]]
    axs[1, 0].streamplot(x, y, u.T, v.T, density=1.5, color='black', linewidth=0.5)
    im2 = axs[1, 0].imshow(mask, cmap='gray', alpha=0.3)
    axs[1, 0].set_title('Streamlines')
    vorticity = np.zeros_like(u)
    for i in range(1, u.shape[0]-1):
        for j in range(1, u.shape[1]-1):
            vorticity[i, j] = (v[i, j+1] - v[i, j-1]) / 2 - (u[i+1, j] - u[i-1, j]) / 2
    
    im3 = axs[1, 1].imshow(vorticity, cmap='RdBu')
    axs[1, 1].set_title('Vorticity')
    plt.colorbar(im3, ax=axs[1, 1])
    
                                              
    if 'airfoil_type' in case_data:
        plt.suptitle(f"NACA {case_data['airfoil_type']} Airfoil, Re={Re:.0f}", fontsize=16)
    else:
        plt.suptitle(f"Airfoil Flow, Re={Re:.0f}", fontsize=16)
    
    plt.tight_layout()
    
                     
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def split_dataset(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    np.random.seed(seed)
                           
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
                        
    npz_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    np.random.shuffle(npz_files)
    
                             
    n_files = len(npz_files)
    n_train = int(n_files * train_ratio)
    n_val = int(n_files * val_ratio)
    
                 
    train_files = npz_files[:n_train]
    val_files = npz_files[n_train:n_train+n_val]
    test_files = npz_files[n_train+n_val:]
    
                                                             
    print("Splitting dataset into train/val/test sets...")
    for split_name, files, target_dir in [
        ("Training", train_files, train_dir), 
        ("Validation", val_files, val_dir), 
        ("Test", test_files, test_dir)
    ]:
        print(f"Processing {split_name} set: {len(files)} files")
        for file in tqdm(files, desc=f"Moving {split_name} files"):
            src_path = os.path.join(data_dir, file)
            dst_path = os.path.join(target_dir, file)
                                                                        
            data = np.load(src_path)
            np.savez_compressed(dst_path, **dict(data))
            os.remove(src_path)                        
    
    print(f"Dataset split: {len(train_files)} training, {len(val_files)} validation, {len(test_files)} test")


def normalize_dataset(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    train_files = [f for f in os.listdir(train_dir) if f.endswith('.npz')]
    u_sum, v_sum, p_sum = 0.0, 0.0, 0.0
    u_sq_sum, v_sq_sum, p_sq_sum = 0.0, 0.0, 0.0
    count = 0
    
    for f in tqdm(train_files, desc="Computing normalization statistics"):
        try:
            data_path = os.path.join(train_dir, f)
            data = np.load(data_path)
            required_keys = ['u', 'v', 'p', 'mask']
            if not all(key in data for key in required_keys):
                print(f"Warning: File {f} is missing one or more required keys {required_keys}. Skipping...")
                continue
            mask = data['mask']
            u = data['u'] * mask
            v = data['v'] * mask
            p = data['p'] * mask
        except Exception as e:
            print(f"Error processing file {f}: {str(e)}. Skipping...")
            continue
        
        u_sum += np.sum(u)
        v_sum += np.sum(v)
        p_sum += np.sum(p)
        
        u_sq_sum += np.sum(u**2)
        v_sq_sum += np.sum(v**2)
        p_sq_sum += np.sum(p**2)
        count += np.sum(mask)
    count = max(count, 1)
    
    means = np.array([u_sum, v_sum, p_sum]) / count
    u_mean, v_mean, p_mean = means
    
                                                              
    vars = np.array([u_sq_sum, v_sq_sum, p_sq_sum]) / count - means**2
    u_std, v_std, p_std = np.sqrt(vars)
    
                                  
    stats = {
        'u_mean': u_mean,
        'u_std': u_std,
        'v_mean': v_mean,
        'v_std': v_std,
        'p_mean': p_mean,
        'p_std': p_std
    }
    
                     
    np.savez(os.path.join(data_dir, 'normalization_stats.npz'), **stats)
    
                   
    print("\nDataset Statistics:")
    print(f"  Velocity u: mean={u_mean:.4f}, std={u_std:.4f}")
    print(f"  Velocity v: mean={v_mean:.4f}, std={v_std:.4f}")
    print(f"  Pressure p: mean={p_mean:.4f}, std={p_std:.4f}")
    
    return stats


def get_gpu_memory_usage():
    if not torch.cuda.is_available():
        return "CUDA not available"
    
    result = []
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            device = torch.cuda.current_device()
            name = torch.cuda.get_device_name(device)
            allocated = torch.cuda.memory_allocated(device) / 1e9
            reserved = torch.cuda.memory_reserved(device) / 1e9
            max_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
            
            result.append({
                "id": i,
                "name": name,
                "allocated_GB": allocated,
                "reserved_GB": reserved,
                "total_GB": max_memory,
                "utilization %": torch.cuda.utilization(device) if hasattr(torch.cuda, 'utilization') else 'N/A'
            })
    
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate CFD dataset with optimized GPU acceleration')
    parser.add_argument('--output_dir', type=str, default='./data', help='Output directory')
    parser.add_argument('--re_min', type=float, default=1000, help='Minimum Reynolds number')
    parser.add_argument('--re_max', type=float, default=100000, help='Maximum Reynolds number')
    parser.add_argument('--n_samples', type=int, default=100, help='Number of samples to generate')
    parser.add_argument('--grid_size', type=int, default=128, help='Grid size (assumed square)')
    parser.add_argument('--airfoil_types', type=str, nargs='+', default=['0012'], 
                       help='List of NACA 4-digit airfoil types')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--progressive', action='store_true', help='Progressively increase Reynolds number')
    parser.add_argument('--max_workers', type=int, default=None, help='Maximum number of parallel workers')
    parser.add_argument('--batch_size', type=int, default=None, help='Process this many simulations per batch')
    parser.add_argument('--max_re_start', type=int, default=None, 
                       help='If specified, limits the maximum Reynolds number for initial testing')
    parser.add_argument('--optimization_level', type=int, default=3, choices=[1, 2, 3],
                       help='GPU optimization level (1=basic, 2=intermediate, 3=aggressive)')
    parser.add_argument('--no-resume', dest='resume', action='store_false', 
                        help='Start from scratch, ignoring any previous checkpoints')
    parser.set_defaults(resume=True)
    
    args = parser.parse_args()
    
                          
    if torch.cuda.is_available():
        print("=" * 50)
        print("GPU INFORMATION")
        print("=" * 50)
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  Compute capability: {props.major}.{props.minor}")
            print(f"  Total memory: {props.total_memory / 1e9:.2f} GB")
            print(f"  Multi processors: {props.multi_processor_count}")
        print("=" * 50)
        
                                
        if args.optimization_level == 1:
            print("Using basic optimization settings")
        elif args.optimization_level == 2:
            print("Using intermediate optimization settings")
            torch.backends.cudnn.benchmark = True
        else:           
            print("Using aggressive optimization settings")
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True
    
                                  
    os.makedirs(args.output_dir, exist_ok=True)
    
                                                                     
    re_max = args.re_max
    if args.max_re_start is not None:
        re_max = min(args.max_re_start, re_max)
        print(f"Limiting maximum Reynolds number to {re_max} for initial testing")
    
                      
    print(f"Generating {args.n_samples} CFD simulations with optimized GPU acceleration...")
    start_time = time.time() if 'time' in globals() else __import__('time').time()
    
    successful_cases = create_dataset_gpu(
        output_dir=args.output_dir,
        re_min=args.re_min,
        re_max=re_max,
        n_samples=args.n_samples,
        grid_size=(args.grid_size, args.grid_size),
        airfoil_types=args.airfoil_types,
        seed=args.seed,
        progressive=args.progressive,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        resume=args.resume
    )
    
                          
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"Data generation completed in {int(hours)}h {int(minutes)}m {seconds:.1f}s")
    print(f"Average time per case: {total_time / max(len(successful_cases), 1):.2f} seconds")
    
                   
    split_dataset(args.output_dir)
    
    stats = normalize_dataset(args.output_dir)
    
    print("\nDataset generation process complete!")
    print(f"Successfully generated {len(successful_cases)} cases")
    print(f"Data saved to: {os.path.abspath(args.output_dir)}")
