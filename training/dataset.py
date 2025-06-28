import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class AirfoilFlowDataset(Dataset):
    """
    Enhanced dataset for loading airfoil flow data for FNO training with data augmentation.
    
    Each sample consists of:
    - Input: Reynolds number and airfoil mask
    - Target: u, v velocity components and pressure field
    
    Features:
    - Normalization of input/output fields
    - Data augmentation (rotation, flipping) for training set
    - Multi-resolution support
    """
    
    def __init__(self, data_dir, split='train', normalize=True, stats_file=None, augment=True):
        """
        Initialize the dataset.
        
        Parameters:
        - data_dir: Base directory containing the dataset
        - split: 'train', 'val', or 'test'
        - normalize: Whether to normalize the data
        - stats_file: Path to normalization statistics file, if None uses {data_dir}/normalization_stats.npz
        """
        self.data_dir = os.path.join(data_dir, split)
        self.split = split
        self.normalize = normalize
        self.augment = augment and split == 'train'                              
        
                                    
        self.file_list = [f for f in os.listdir(self.data_dir) if f.endswith('.npz') and f.startswith('case_')]
        self.file_list.sort()                              
        
                                                 
        if normalize:
            if stats_file is None:
                                                                                                     
                stats_file_in_data_dir = os.path.join(data_dir, 'normalization_stats.npz')
                stats_file_in_parent = os.path.join(os.path.dirname(data_dir), 'normalization_stats.npz')
                
                if os.path.exists(stats_file_in_data_dir):
                    stats_file = stats_file_in_data_dir
                else:
                    stats_file = stats_file_in_parent
            
            if os.path.exists(stats_file):
                self.stats = dict(np.load(stats_file))
            else:
                raise FileNotFoundError(f"Normalization statistics file not found: {stats_file}")
    
    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.file_list)
    
    def apply_augmentation(self, inputs, targets, mask):
        """
        Apply data augmentation to improve model generalization
        
        Parameters:
        - inputs: Input tensor [channels, height, width]
        - targets: Target tensor [channels, height, width]
        - mask: Airfoil mask [height, width]
        
        Returns:
        - Augmented inputs, targets, mask
        """
                                                  
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs)
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets)
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
            
                                                              
        if torch.rand(1).item() < 0.5:
                                                 
            k = torch.randint(0, 4, (1,)).item()
            inputs = torch.rot90(inputs, k, dims=[-2, -1])
            targets = torch.rot90(targets, k, dims=[-2, -1])
            mask = torch.rot90(mask, k, dims=[-2, -1])
            
                                                                  
            if k % 2 == 1:                     
                                                                  
                targets[0], targets[1] = targets[1].clone(), -targets[0].clone()
        
                                
        if torch.rand(1).item() < 0.5:
            inputs = torch.flip(inputs, [-1])
            targets = torch.flip(targets, [-1])
            mask = torch.flip(mask, [-1])
                                                               
            targets[0] = -targets[0]
            
                              
        if torch.rand(1).item() < 0.5:
            inputs = torch.flip(inputs, [-2])
            targets = torch.flip(targets, [-2])
            mask = torch.flip(mask, [-2])
                                                             
            targets[1] = -targets[1]
            
        return inputs, targets, mask
        
    def __getitem__(self, idx):
        """Get a sample from the dataset with optional augmentation"""
                             
        data_path = os.path.join(self.data_dir, self.file_list[idx])
        data = np.load(data_path)
        
                        
        u = data['u'].astype(np.float32)
        v = data['v'].astype(np.float32)
        p = data['p'].astype(np.float32)
        mask = data['mask'].astype(np.float32)
        Re = data['Re'].astype(np.float32)
        
                          
        if self.normalize:
            u = (u - self.stats['u_mean']) / self.stats['u_std']
            v = (v - self.stats['v_mean']) / self.stats['v_std']
            p = (p - self.stats['p_mean']) / self.stats['p_std']
        
                                                       
                                            
        Re_normalized = (np.log10(Re) - 3) / 2
        
                                                                
        Re_channel = np.ones_like(mask) * Re_normalized
        
                              
        inputs = np.stack([Re_channel, mask], axis=0)
        
                               
        targets = np.stack([u, v, p], axis=0)
        
                                                         
        targets[0] *= mask              
        targets[1] *= mask              
        targets[2] *= mask            
        
                                    
        inputs = torch.from_numpy(inputs)
        targets = torch.from_numpy(targets)
        mask = torch.from_numpy(mask)
        
                                            
        if self.augment:
            inputs, targets, mask = self.apply_augmentation(inputs, targets, mask)
        
                                                                                       
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs.astype(np.float32))
        else:                    
            inputs = inputs.to(dtype=torch.float32)
            
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets.astype(np.float32))
        else:                    
            targets = targets.to(dtype=torch.float32)
            
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask.astype(np.float32))
        else:                    
            mask = mask.to(dtype=torch.float32)
        
        return {'inputs': inputs, 'targets': targets, 'mask': mask}


def get_dataloader(data_dir, batch_size=16, split='train', normalize=True, augment=True, num_workers=4, **kwargs):
    """
    Create a DataLoader for the airfoil flow dataset.
    
    Parameters:
    - data_dir: Base directory containing the dataset
    - batch_size: Batch size
    - split: 'train', 'val', or 'test'
    - normalize: Whether to normalize the data
    - augment: Whether to use data augmentation (only applies to training set)
    - num_workers: Number of worker processes
    - **kwargs: Additional arguments for DataLoader
    
    Returns:
    - dataloader: PyTorch DataLoader
    """
    dataset = AirfoilFlowDataset(data_dir, split, normalize, augment=augment)
    
                                                         
    loader_args = {
        'batch_size': batch_size,
        'shuffle': (split == 'train'),
        'num_workers': num_workers,
        'pin_memory': True
    }
    
                                          
    loader_args.update(kwargs)
    
    dataloader = DataLoader(dataset, **loader_args)
    
    return dataloader


if __name__ == "__main__":
                          
    import matplotlib.pyplot as plt
    
                                   
    data_dir = '../data'
    dataset = AirfoilFlowDataset(data_dir, split='train')
    
    sample = dataset[0]
    
                      
    print(f"Input shape: {sample['inputs'].shape}")
    print(f"Target shape: {sample['targets'].shape}")
    
               
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
                                      
    axes[0, 0].imshow(sample['inputs'][0].numpy(), cmap='viridis')
    axes[0, 0].set_title('Reynolds Number Channel')
    
    axes[0, 1].imshow(sample['inputs'][1].numpy(), cmap='gray')
    axes[0, 1].set_title('Airfoil Mask')
    
                                     
    axes[1, 0].imshow(sample['targets'][0].numpy(), cmap='RdBu_r')
    axes[1, 0].set_title('U-Velocity')
    
    axes[1, 1].imshow(sample['targets'][2].numpy(), cmap='coolwarm')
    axes[1, 1].set_title('Pressure')
    
    plt.tight_layout()
    plt.savefig('dataset_sample.png')
    plt.show()
