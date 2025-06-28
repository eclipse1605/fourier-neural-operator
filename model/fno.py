import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpectralConv2d(nn.Module):
    """
    Enhanced 2D Fourier layer with comprehensive spectral domain handling.
    Implements FFT -> Multiplication by weights -> IFFT with full frequency domain coverage.
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1                                                          
        self.modes2 = modes2                                                           
        
                                                                      
                                         
        self.scale = 1 / (in_channels * out_channels)
        
                                                           
                                           
        self.weights1 = nn.Parameter(
            self.scale * torch.complex(
                torch.randn(in_channels, out_channels, self.modes1, self.modes2),
                torch.randn(in_channels, out_channels, self.modes1, self.modes2)
            )
        )
        
                                                                      
                                                                                            
        self.weights2 = nn.Parameter(
            self.scale * torch.complex(
                torch.randn(in_channels, out_channels, self.modes1, self.modes2),
                torch.randn(in_channels, out_channels, self.modes1, self.modes2)
            )
        )

    def compl_mul2d(self, input, weights):
        """
        Complex multiplication between input and weights in spectral domain.
        Optimized implementation using einsum for better performance.
        
        Args:
            input: Complex tensor in spectral domain
            weights: Complex weights
            
        Returns:
            Complex tensor after multiplication
        """
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        """
        Forward pass with improved implementation for spectral convolution.
        
        Args:
            x: Input tensor [batch, channels, height, width]
            
        Returns:
            Output tensor after spectral convolution
        """
        batchsize = x.shape[0]
        
                                      
        x_ft = torch.fft.rfft2(x)
        
                                                         
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, 
                             dtype=torch.cfloat, device=x.device)
        
                                                                       
                                                               
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2],
            self.weights1
        )
        
                                                                                              
                                                                                               
        
                                          
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        
        return x


class FNO2d(nn.Module):
    """
    Enhanced Fourier Neural Operator for 2D problems with:
    - Improved spectral convolution
    - Batch normalization support
    - Dropout for uncertainty quantification
    - Resolution invariance by design
    
    Parameters:
    - in_channels: Number of input channels (e.g., 1 for just velocity magnitude)
    - out_channels: Number of output channels (e.g., 3 for [u, v, p] fields)
    - width: Width of the network, number of channels in hidden layers
    - modes1, modes2: Number of Fourier modes to keep in each dimension
    - n_layers: Number of Fourier layers
    - device: Device to place the model on ('cuda', 'cuda:0', 'cpu', etc.)
    - use_bn: Whether to use batch normalization
    - dropout_rate: Dropout rate for uncertainty quantification (0 to disable)
    """
    def __init__(self, in_channels=1, out_channels=3, width=32, modes1=12, modes2=12, 
                 n_layers=4, device=None, use_bn=True, dropout_rate=0.0):
        super(FNO2d, self).__init__()
        
                         
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
                          
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.modes1 = modes1
        self.modes2 = modes2
        self.n_layers = n_layers
        self.use_bn = use_bn
        self.dropout_rate = dropout_rate
        
                                
        self.fc0 = nn.Linear(in_channels, self.width)
        
                        
        self.fourier_layers = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_bn else None
        
        for i in range(self.n_layers):
            self.fourier_layers.append(
                SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
            )
            
                                               
            self.convs.append(nn.Conv2d(self.width, self.width, kernel_size=1))
            
                                               
            if use_bn:
                self.batch_norms.append(nn.BatchNorm2d(self.width))
        
                                 
        self.fc1 = nn.Linear(self.width, out_channels)
        
                             
        self.activation = nn.GELU()
        
                                            
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        
                              
        self.to(self.device)
    
    def forward(self, x, return_uncertainty=False, n_samples=10):
        """
        Enhanced forward pass of the FNO model with support for uncertainty quantification
        
        Parameters:
        - x: Input tensor of shape [batch_size, in_channels, height, width]
        - return_uncertainty: Whether to return prediction uncertainty (requires dropout > 0)
        - n_samples: Number of Monte Carlo samples for uncertainty estimation
        
        Returns:
        - Output tensor of shape [batch_size, out_channels, height, width] if not return_uncertainty
        - Tuple of (mean, std) tensors if return_uncertainty is True
        """
                                        
        x = x.to(self.device)
        
                                                        
        batchsize = x.shape[0]
        size_x, size_y = x.shape[2], x.shape[3]
        
                                                        
        if return_uncertainty and self.dropout_rate > 0:
            return self._forward_with_uncertainty(x, n_samples)
        
                               
                                                                   
        x = x.permute(0, 2, 3, 1)                           
        x = self.fc0(x)                                        
        x = x.permute(0, 3, 1, 2)                        
        
                                                       
        for i in range(self.n_layers):
                                  
            x1 = self.fourier_layers[i](x)
            
                                     
            x2 = self.convs[i](x)
            
                                            
            x = x1 + x2
            
                                                  
            if self.use_bn:
                x = self.batch_norms[i](x)
                
                              
            x = self.activation(x)
            
                                                             
            if self.dropout is not None and self.training:
                x = self.dropout(x)
        
                                       
        x = x.permute(0, 2, 3, 1)                        
        x = self.fc1(x)                                                  
        x = x.permute(0, 3, 1, 2)                               
        
        return x
        
    def _forward_with_uncertainty(self, x, n_samples=10):
        """
        Forward pass with Monte Carlo dropout for uncertainty quantification
        
        Parameters:
        - x: Input tensor
        - n_samples: Number of Monte Carlo samples
        
        Returns:
        - mean: Mean prediction
        - std: Standard deviation of predictions
        """
                                         
        was_training = self.training
        self.training = True
        
                                       
        outputs = []
        
                                                    
        with torch.no_grad():
            for _ in range(n_samples):
                outputs.append(self.forward(x))
        
                                                  
        outputs = torch.stack(outputs, dim=0)
        
                                               
        mean = torch.mean(outputs, dim=0)
        std = torch.std(outputs, dim=0)
        
                                                                 
        self.training = was_training
        
        return mean, std
    
    def count_params(self):
        """Count total parameters in the model"""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params
    
    def to_device(self, device):
        """Move model to specific device"""
        self.device = torch.device(device)
        self.to(self.device)
        return self
    
    def analyze_modes(self, input_data, layer_idx=0, top_k=5):
        """
        Analyze which Fourier modes are most important for this model's predictions.
        
        Parameters:
        - input_data: Sample input tensor [batch, channels, height, width]
        - layer_idx: Which Fourier layer to analyze (default: first layer)
        - top_k: Number of top modes to report
        
        Returns:
        - Dictionary with mode analysis results
        """
                                      
        was_training = self.training
        self.eval()
        
        with torch.no_grad():
                                       
            x = input_data.to(self.device)
            
                                                          
            x = x.permute(0, 2, 3, 1)                           
            x = self.fc0(x)                                        
            x = x.permute(0, 3, 1, 2)                        
            
                                             
            for i in range(layer_idx):
                x1 = self.fourier_layers[i](x)
                x2 = self.convs[i](x)
                x = self.activation(x1 + x2)
                if self.use_bn:
                    x = self.batch_norms[i](x)
            
                                       
            x_ft = torch.fft.rfft2(x)
            
                                        
            weights = self.fourier_layers[layer_idx].weights1
            
                                                    
            weight_magnitude = torch.abs(weights).mean(dim=(0, 1))                                  
            
                              
            if top_k > weight_magnitude.numel():
                top_k = weight_magnitude.numel()
                
            values, indices = torch.topk(weight_magnitude.flatten(), top_k)
            top_modes = [(idx // weight_magnitude.shape[1], idx % weight_magnitude.shape[1]) 
                        for idx in indices.cpu().numpy()]
            
                                 
            self.train(was_training)
            
            return {
                "top_k_values": values.cpu().numpy(),
                "top_k_modes": top_modes,
                "weights_shape": list(weight_magnitude.shape),
                "layer_idx": layer_idx
            }
    
    def test_resolution_invariance(self, input_data, scale_factors=[0.5, 1.0, 2.0]):
        """
        Test the resolution invariance property of the FNO model.
        
        Parameters:
        - input_data: Sample input tensor [batch, channels, height, width]
        - scale_factors: List of scaling factors to test
        
        Returns:
        - Dictionary with resolution invariance test results
        """
        results = {}
        original_shape = input_data.shape
        was_training = self.training
        self.eval()
        
        with torch.no_grad():
                                     
            original_output = self.forward(input_data).cpu()
            
            for factor in scale_factors:
                if factor == 1.0:
                    continue
                    
                                                  
                new_h = int(original_shape[2] * factor)
                new_w = int(original_shape[3] * factor)
                resized_input = F.interpolate(
                    input_data, 
                    size=(new_h, new_w), 
                    mode='bilinear', 
                    align_corners=False
                )
                
                                                  
                resized_output = self.forward(resized_input)
                
                                                                   
                resized_output = F.interpolate(
                    resized_output, 
                    size=(original_shape[2], original_shape[3]), 
                    mode='bilinear', 
                    align_corners=False
                ).cpu()
                
                                           
                error = torch.norm(resized_output - original_output) / torch.norm(original_output)
                results[factor] = error.item()
        
                             
        self.train(was_training)        
        return results


def divergence(velocity_field, mask=None, higher_order=False):
    """
    Compute the divergence of a 2D velocity field with optional higher-order accuracy.
    Used for incompressibility constraint in physics-informed loss functions.
    
    Parameters:
    - velocity_field: Tensor of shape [batch_size, 2, height, width] where the first channel is u and second is v
    - mask: Optional binary mask for the airfoil (1 for fluid, 0 for solid)
    - higher_order: Whether to use 4th-order central differencing
    
    Returns:
    - div: Divergence of the velocity field
    """
    device = velocity_field.device
    batch_size = velocity_field.shape[0]
    u = velocity_field[:, 0]                           
    v = velocity_field[:, 1]                           
    
                                 
    du_dx = torch.zeros_like(u, device=device)
    dv_dy = torch.zeros_like(v, device=device)
    
    if higher_order and u.shape[-1] > 4 and u.shape[-2] > 4:
                                                            
        du_dx[:, 2:-2, 2:-2] = (-u[:, 2:-2, 4:] + 8*u[:, 2:-2, 3:-1] - 8*u[:, 2:-2, 1:-3] + u[:, 2:-2, :-4]) / 12.0
        dv_dy[:, 2:-2, 2:-2] = (-v[:, 4:, 2:-2] + 8*v[:, 3:-1, 2:-2] - 8*v[:, 1:-3, 2:-2] + v[:, :-4, 2:-2]) / 12.0
        
                                            
        du_dx[:, 1:-1, [1, -2]] = (u[:, 1:-1, [2, -1]] - u[:, 1:-1, [0, -3]]) / 2.0
        dv_dy[:, [1, -2], 1:-1] = (v[:, [2, -1], 1:-1] - v[:, [0, -3], 1:-1]) / 2.0
        
                                       
        du_dx[:, :, 0] = u[:, :, 1] - u[:, :, 0]
        du_dx[:, :, -1] = u[:, :, -1] - u[:, :, -2]
        dv_dy[:, 0, :] = v[:, 1, :] - v[:, 0, :]
        dv_dy[:, -1, :] = v[:, -1, :] - v[:, -2, :]
    else:
                                                              
        du_dx[:, 1:-1, 1:-1] = (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) / 2.0
        dv_dy[:, 1:-1, 1:-1] = (v[:, 2:, 1:-1] - v[:, :-2, 1:-1]) / 2.0
        
                                       
        du_dx[:, :, 0] = u[:, :, 1] - u[:, :, 0]
        du_dx[:, :, -1] = u[:, :, -1] - u[:, :, -2]
        dv_dy[:, 0, :] = v[:, 1, :] - v[:, 0, :]
        dv_dy[:, -1, :] = v[:, -1, :] - v[:, -2, :]
    
                        
    div = du_dx + dv_dy
    
                            
    if mask is not None:
                                              
        if mask.device != device:
            mask = mask.to(device)
        div = div * mask
    
    return div
