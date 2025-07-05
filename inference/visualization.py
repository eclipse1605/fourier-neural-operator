import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.animation as animation
from matplotlib.cm import ScalarMappable
import torch
from pathlib import Path
import sys

                                                        
sys.path.append(str(Path(__file__).parent.parent))

class FlowVisualizer:
    def __init__(self, save_dir='./results/visualizations'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
                             
        self.figsize = (18, 10)
        
                           
        self.velocity_cmap = 'viridis'
        self.pressure_cmap = 'plasma'
        self.error_cmap = 'hot'
        self.uncertainty_cmap = 'inferno'
    
    def _to_numpy(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor
    
    def plot_field_comparison(self, prediction, target, mask, index=0, 
                             show=True, save=False, filename=None):
        prediction = self._to_numpy(prediction)
        target = self._to_numpy(target)
        mask = self._to_numpy(mask)
        
                               
        if prediction.ndim > 3:
            prediction = prediction[index]
        if target.ndim > 3:
            target = target[index]
        if mask.ndim > 2:
            mask = mask[index]
        
                    
        prediction_masked = [prediction[i] * mask for i in range(prediction.shape[0])]
        target_masked = [target[i] * mask for i in range(target.shape[0])]
        
                         
        error = [np.abs(prediction_masked[i] - target_masked[i]) for i in range(prediction.shape[0])]
        
                       
        fig, axs = plt.subplots(3, 4, figsize=self.figsize)
        fig.suptitle('Flow Field Comparison: Prediction vs Ground Truth', fontsize=16)
        
                                     
        channels = ['u-velocity', 'v-velocity', 'pressure']
        cmaps = [self.velocity_cmap, self.velocity_cmap, self.pressure_cmap]
        
                           
        for i, (channel, cmap) in enumerate(zip(channels, cmaps)):
                                                       
            vmin = min(np.min(prediction_masked[i]), np.min(target_masked[i]))
            vmax = max(np.max(prediction_masked[i]), np.max(target_masked[i]))
            norm = Normalize(vmin=vmin, vmax=vmax)
            
                             
            im = axs[i, 0].imshow(prediction_masked[i], cmap=cmap, norm=norm)
            axs[i, 0].set_title(f'Predicted {channel}')
            plt.colorbar(im, ax=axs[i, 0])
            
                               
            im = axs[i, 1].imshow(target_masked[i], cmap=cmap, norm=norm)
            axs[i, 1].set_title(f'Ground Truth {channel}')
            plt.colorbar(im, ax=axs[i, 1])
            
                             
            im = axs[i, 2].imshow(error[i], cmap=self.error_cmap)
            axs[i, 2].set_title(f'Absolute Error - {channel}')
            plt.colorbar(im, ax=axs[i, 2])
            
                                                                   
            rel_error = np.zeros_like(error[i])
            nonzero_idx = np.abs(target_masked[i]) > 1e-6
            rel_error[nonzero_idx] = error[i][nonzero_idx] / np.abs(target_masked[i][nonzero_idx])
            im = axs[i, 3].imshow(np.clip(rel_error, 0, 1), cmap=self.error_cmap, vmax=0.5)
            axs[i, 3].set_title(f'Relative Error - {channel}')
            plt.colorbar(im, ax=axs[i, 3])
        
        plt.tight_layout()
        
                     
        if save:
            filename = filename or f'field_comparison_{index}.png'
            plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
            
        return fig
    
    def plot_uncertainty(self, mean, std, mask, index=0, 
                        show=True, save=False, filename=None):
        """
        Plot uncertainty visualization
        
        Parameters:
        - mean: Mean prediction tensor [batch, channels, height, width]
        - std: Standard deviation tensor [batch, channels, height, width]
        - mask: Mask tensor [batch, height, width]
        - index: Index of the sample to visualize
        - show: Whether to display the plot
        - save: Whether to save the plot
        - filename: Filename to save the plot
        """
                          
        mean = self._to_numpy(mean)
        std = self._to_numpy(std)
        mask = self._to_numpy(mask)
        
                               
        if mean.ndim > 3:
            mean = mean[index]
        if std.ndim > 3:
            std = std[index]
        if mask.ndim > 2:
            mask = mask[index]
        
                    
        mean_masked = [mean[i] * mask for i in range(mean.shape[0])]
        std_masked = [std[i] * mask for i in range(std.shape[0])]
        
                                                                       
        cv = []
        for i in range(mean.shape[0]):
            cv_i = np.zeros_like(mean_masked[i])
            nonzero_idx = np.abs(mean_masked[i]) > 1e-6
            cv_i[nonzero_idx] = std_masked[i][nonzero_idx] / np.abs(mean_masked[i][nonzero_idx])
            cv.append(cv_i)
        
                       
        fig, axs = plt.subplots(3, 3, figsize=self.figsize)
        fig.suptitle('Uncertainty Quantification with Monte Carlo Dropout', fontsize=16)
        
                                     
        channels = ['u-velocity', 'v-velocity', 'pressure']
        cmaps = [self.velocity_cmap, self.velocity_cmap, self.pressure_cmap]
        
                           
        for i, (channel, cmap) in enumerate(zip(channels, cmaps)):
                                  
            im = axs[i, 0].imshow(mean_masked[i], cmap=cmap)
            axs[i, 0].set_title(f'Mean {channel}')
            plt.colorbar(im, ax=axs[i, 0])
            
                                     
            im = axs[i, 1].imshow(std_masked[i], cmap=self.uncertainty_cmap)
            axs[i, 1].set_title(f'Standard Deviation {channel}')
            plt.colorbar(im, ax=axs[i, 1])
            
                                                         
            im = axs[i, 2].imshow(np.clip(cv[i], 0, 1), cmap=self.uncertainty_cmap, vmax=0.5)
            axs[i, 2].set_title(f'Coefficient of Variation {channel}')
            plt.colorbar(im, ax=axs[i, 2])
        
        plt.tight_layout()
        
                     
        if save:
            filename = filename or f'uncertainty_{index}.png'
            plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
            
        return fig
    
    def plot_streamlines(self, u, v, mask, prediction=True, target=None, 
                        density=2, index=0, show=True, save=False, filename=None):
                          
        u = self._to_numpy(u)
        v = self._to_numpy(v)
        mask = self._to_numpy(mask)
        
                                        
        if u.ndim > 2:
            u = u[index]
        if v.ndim > 2:
            v = v[index]
        if mask.ndim > 2:
            mask = mask[index]
        
                    
        u_masked = u * mask
        v_masked = v * mask
        
                                     
        y, x = np.mgrid[0:u.shape[0], 0:u.shape[1]]
        
                            
        speed = np.sqrt(u_masked**2 + v_masked**2)
        
                       
        if target is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            title = 'Predicted Flow' if prediction else 'Ground Truth Flow'
            fig.suptitle(title, fontsize=16)
            
                                                   
            im = ax.imshow(speed, cmap='viridis')
            plt.colorbar(im, ax=ax, label='Velocity Magnitude')
            
                              
            ax.streamplot(x, y, u_masked, v_masked, density=density, color='white')
            
        else:
                                                  
            u_target, v_target = target
            u_target = self._to_numpy(u_target)
            v_target = self._to_numpy(v_target)
            
                                            
            if u_target.ndim > 2:
                u_target = u_target[index]
            if v_target.ndim > 2:
                v_target = v_target[index]
                
                        
            u_target_masked = u_target * mask
            v_target_masked = v_target * mask
            
                              
            speed_target = np.sqrt(u_target_masked**2 + v_target_masked**2)
            
                                           
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('Flow Streamlines: Prediction vs Ground Truth', fontsize=16)
            
                                                 
            vmin = min(np.min(speed), np.min(speed_target))
            vmax = max(np.max(speed), np.max(speed_target))
            norm = Normalize(vmin=vmin, vmax=vmax)
            
                                 
            im = axs[0].imshow(speed, cmap='viridis', norm=norm)
            axs[0].streamplot(x, y, u_masked, v_masked, density=density, color='white')
            axs[0].set_title('Predicted Flow')
            plt.colorbar(im, ax=axs[0], label='Velocity Magnitude')
            
                                    
            im = axs[1].imshow(speed_target, cmap='viridis', norm=norm)
            axs[1].streamplot(x, y, u_target_masked, v_target_masked, density=density, color='white')
            axs[1].set_title('Ground Truth Flow')
            plt.colorbar(im, ax=axs[1], label='Velocity Magnitude')
            
                             
            error = np.sqrt((u_masked - u_target_masked)**2 + (v_masked - v_target_masked)**2)
            im = axs[2].imshow(error, cmap='hot')
            axs[2].set_title('Velocity Error Magnitude')
            plt.colorbar(im, ax=axs[2], label='Error Magnitude')
        
        plt.tight_layout()
        
                     
        if save:
            filename = filename or f'streamlines_{index}.png'
            plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
            
        return fig
    
    def plot_physical_correctness(self, u, v, mask, index=0, h=1.0,
                                show=True, save=False, filename=None):
        u = self._to_numpy(u)
        v = self._to_numpy(v)
        mask = self._to_numpy(mask)
        
                               
        if u.ndim > 2:
            u = u[index]
        if v.ndim > 2:
            v = v[index]
        if mask.ndim > 2:
            mask = mask[index]
        
                    
        u_masked = u * mask
        v_masked = v * mask
        
                                                      
        du_dx = np.zeros_like(u_masked)
        dv_dy = np.zeros_like(v_masked)
        
                                                
        du_dx[:, 1:-1] = (u_masked[:, 2:] - u_masked[:, :-2]) / (2 * h)
        
                                                
        dv_dy[1:-1, :] = (v_masked[2:, :] - v_masked[:-2, :]) / (2 * h)
        
                    
        divergence = du_dx + dv_dy
        
                                                           
        divergence = divergence * mask
        
                       
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Physical Correctness: Divergence Analysis', fontsize=16)
        
                                  
        im = axs[0].imshow(u_masked, cmap='viridis')
        axs[0].set_title('u-velocity')
        plt.colorbar(im, ax=axs[0])
        
        im = axs[1].imshow(v_masked, cmap='viridis')
        axs[1].set_title('v-velocity')
        plt.colorbar(im, ax=axs[1])
        
                         
        div_max = np.max(np.abs(divergence))
        im = axs[2].imshow(divergence, cmap='seismic', vmin=-div_max, vmax=div_max)
        axs[2].set_title('Divergence (∇·u)')
        plt.colorbar(im, ax=axs[2])
        
                                   
        div_mean = np.mean(np.abs(divergence[mask > 0.5]))
        div_max = np.max(np.abs(divergence[mask > 0.5]))
        axs[2].text(0.05, 0.95, f'Mean |∇·u|: {div_mean:.6f}', transform=axs[2].transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axs[2].text(0.05, 0.85, f'Max |∇·u|: {div_max:.6f}', transform=axs[2].transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
                     
        if save:
            filename = filename or f'divergence_{index}.png'
            plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
            
        return fig, divergence
    
    def visualize_test_case(self, prediction, target, mask, index=0,
                           show=True, save=True, save_dir=None):
        if save_dir is not None:
            orig_save_dir = self.save_dir
            self.save_dir = save_dir
            os.makedirs(save_dir, exist_ok=True)
        
                                                              
        pred = self._to_numpy(prediction)
        targ = self._to_numpy(target)
        msk = self._to_numpy(mask)
        
        if pred.ndim > 3:
            pred = pred[index]
        if targ.ndim > 3:
            targ = targ[index]
        if msk.ndim > 2:
            msk = msk[index]
        
                             
        self.plot_field_comparison(pred, targ, msk, 
                                 show=show, save=save, 
                                 filename=f'case_{index}_fields.png')
        
                        
        self.plot_streamlines(pred[0], pred[1], msk, 
                            prediction=True, target=(targ[0], targ[1]),
                            show=show, save=save, 
                            filename=f'case_{index}_streamlines.png')
        
                                 
        self.plot_physical_correctness(pred[0], pred[1], msk,
                                     show=show, save=save,
                                     filename=f'case_{index}_divergence.png')
                                     
                                         
        if save_dir is not None:
            self.save_dir = orig_save_dir
            
    def visualize_uncertainty_analysis(self, mean, std, target, mask, index=0,
                                     show=True, save=True, save_dir=None):
        if save_dir is not None:
            orig_save_dir = self.save_dir
            self.save_dir = save_dir
            os.makedirs(save_dir, exist_ok=True)
        
                                      
        self.plot_uncertainty(mean, std, mask, index=index,
                            show=show, save=save,
                            filename=f'case_{index}_uncertainty.png')
        
                                                  
        self.plot_field_comparison(mean, target, mask, index=index,
                                 show=show, save=save,
                                 filename=f'case_{index}_mean_comparison.png')
                                 
                                             
        mn = self._to_numpy(mean)
        targ = self._to_numpy(target)
        msk = self._to_numpy(mask)
        
        if mn.ndim > 3:
            mn = mn[index]
        if targ.ndim > 3:
            targ = targ[index]
        if msk.ndim > 2:
            msk = msk[index]
            
        self.plot_streamlines(mn[0], mn[1], msk,
                            prediction=True, target=(targ[0], targ[1]),
                            show=show, save=save,
                            filename=f'case_{index}_mean_streamlines.png')
                            
                                         
        if save_dir is not None:
            self.save_dir = orig_save_dir


                            
if __name__ == "__main__":
                           
    batch, channels, height, width = 2, 3, 64, 64
    
                             
    prediction = np.random.randn(batch, channels, height, width) * 0.1
    target = np.random.randn(batch, channels, height, width) * 0.1
    
                            
    mask = np.ones((batch, height, width))
    for b in range(batch):
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2
        radius = min(height, width) // 4
        mask[b] = ((x - center_x)**2 + (y - center_y)**2 <= radius**2).astype(float)
    
                           
    vis = FlowVisualizer(save_dir='./test_vis')
    
                           
    vis.plot_field_comparison(prediction, target, mask, index=0, save=True)
    
                      
    vis.plot_streamlines(prediction[0, 0], prediction[0, 1], mask[0], 
                       target=(target[0, 0], target[0, 1]), save=True)
    
                                      
    vis.visualize_test_case(prediction, target, mask, index=0, save=True)
    
    print("Visualization tests completed!")
