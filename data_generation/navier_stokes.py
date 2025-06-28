import torch
import numpy as np
import time
from tqdm import tqdm

class NavierStokesGPUSolver:
    def __init__(self, grid_size=(128, 128), Re=1000, dt=0.001, airfoil_mask=None, device=None):
        self.device = device if device is not None else (
            'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"NavierStokesGPUSolver initialized on: {self.device}")
        
        self.height, self.width = grid_size
        self.Re = Re
        self.dt = dt
        
                                                             
        if airfoil_mask is None:
            self.mask = torch.ones(grid_size, dtype=torch.float32, device=self.device)
        else:
                                                          
            if isinstance(airfoil_mask, np.ndarray):
                self.mask = torch.from_numpy(airfoil_mask).float().to(self.device)
            else:
                self.mask = airfoil_mask.float().to(self.device)
        
                                                                                
        self.fluid_mask = (self.mask > 0).float()
        
                                              
        self.dx = 1.0 / (self.width - 1)
        self.dy = 1.0 / (self.height - 1)
        
                                                                         
        self.dx2 = self.dx * self.dx
        self.dy2 = self.dy * self.dy
        self.factor = self.dx2 * self.dy2 / (2.0 * (self.dx2 + self.dy2))
        
                                                 
        self.u = torch.zeros(grid_size, dtype=torch.float32, device=self.device)
        self.v = torch.zeros(grid_size, dtype=torch.float32, device=self.device)
        self.p = torch.zeros(grid_size, dtype=torch.float32, device=self.device)
        
                                                          
        self.u_prev = torch.zeros_like(self.u)
        self.v_prev = torch.zeros_like(self.v)
        
                                            
        self.nu = 1.0 / Re
        
                                                     
        self.tmp_u = torch.zeros_like(self.u)
        self.tmp_v = torch.zeros_like(self.v)
        self.tmp_p = torch.zeros_like(self.p)
        
    def set_boundary_conditions(self, u_in=1.0, top_wall=True, bottom_wall=True):
        self.u[:, 0] = u_in * self.mask[:, 0]
        self.v[:, 0] = 0.0
        
                                                           
        self.u[:, -1] = self.u[:, -2]
        self.v[:, -1] = self.v[:, -2]
        
                      
        if top_wall:
            self.u[0, :] = 0.0           
            self.v[0, :] = 0.0           
        else:
                                   
            self.u[0, :] = self.u[1, :]
            self.v[0, :] = 0.0
        
                         
        if bottom_wall:
            self.u[-1, :] = 0.0           
            self.v[-1, :] = 0.0           
        else:
                                   
            self.u[-1, :] = self.u[-2, :]
            self.v[-1, :] = 0.0
        
                                                           
        self.u = self.u * self.mask
        self.v = self.v * self.mask

    def advection_term(self, field, u, v):
        u_clamped = torch.clamp(u, -10.0, 10.0)
        v_clamped = torch.clamp(v, -10.0, 10.0)
        u_pos = u_clamped > 0
        v_pos = v_clamped > 0
        
                                      
        df_dx = torch.zeros_like(field)
        df_dy = torch.zeros_like(field)
                           
        df_dx[1:-1, 1:-1] = torch.where(
            u_pos[1:-1, 1:-1],
            (field[1:-1, 1:-1] - field[1:-1, 0:-2]) / self.dx,                 
            (field[1:-1, 2:] - field[1:-1, 1:-1]) / self.dx                   
        )
        df_dy[1:-1, 1:-1] = torch.where(
            v_pos[1:-1, 1:-1],
            (field[1:-1, 1:-1] - field[0:-2, 1:-1]) / self.dy,                 
            (field[2:, 1:-1] - field[1:-1, 1:-1]) / self.dy                   
        )
        adv = u_clamped * df_dx + v_clamped * df_dy
        adv = adv * self.fluid_mask
        return adv

    def diffusion_term(self, field):
        diff = torch.zeros_like(field)
        diff[1:-1, 1:-1] = (
            (field[1:-1, 2:] - 2*field[1:-1, 1:-1] + field[1:-1, 0:-2]) / self.dx2 +
            (field[2:, 1:-1] - 2*field[1:-1, 1:-1] + field[0:-2, 1:-1]) / self.dy2
        )
        diff = diff * self.fluid_mask
        return diff

    def pressure_poisson(self, u, v, max_iter=50, tolerance=1e-5):
        p = torch.zeros_like(self.p)
        
        u_x = torch.zeros_like(u)
        v_y = torch.zeros_like(v)
        
                                                 
        u_x[1:-1, 1:-1] = (u[1:-1, 2:] - u[1:-1, 0:-2]) / (2.0 * self.dx)
        v_y[1:-1, 1:-1] = (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2.0 * self.dy)
        
                                        
        div = torch.clamp(u_x + v_y, -1e3, 1e3) * self.fluid_mask
        
                                                                                 
        for k in range(max_iter):
            p_old = p.clone()
            
                                                                  
            p[1:-1, 1:-1] = self.factor * (
                (p_old[1:-1, 2:] + p_old[1:-1, 0:-2]) / self.dx2 +
                (p_old[2:, 1:-1] + p_old[0:-2, 1:-1]) / self.dy2 -
                div[1:-1, 1:-1]
            )
            
                                                       
            p = p * self.fluid_mask
            
                                                                         
            p[:, 0] = p[:, 1]          
            p[:, -1] = p[:, -2]         
            p[0, :] = p[1, :]         
            p[-1, :] = p[-2, :]          
            
                                                                 
            if torch.norm(p - p_old) < tolerance:
                break
            
                                                                                           
            if k < max_iter - 1:                                 
                                                                   
                no_grad_ctx = torch.no_grad()
                no_grad_ctx.__enter__()
                no_grad_exit = True
        
                                                
                                                            
        if 'no_grad_exit' in locals() and no_grad_exit:
            no_grad_ctx.__exit__(None, None, None)
        
        return p

    def time_step(self):
        self.u_prev.copy_(self.u)
        self.v_prev.copy_(self.v)
                                        
        if (torch.isnan(self.u_prev).any() or 
            torch.isnan(self.v_prev).any() or
            torch.isinf(self.u_prev).any() or
            torch.isinf(self.v_prev).any()):
                                            
            self.dt *= 0.5
            print(f"Reducing time step to {self.dt:.6f} due to instability")
            return False
        
                                                                         
        u_adv = self.advection_term(self.u_prev, self.u_prev, self.v_prev)
        v_adv = self.advection_term(self.v_prev, self.u_prev, self.v_prev)
        
        u_diff = self.diffusion_term(self.u_prev)
        v_diff = self.diffusion_term(self.v_prev)
        
                                                                    
        u_star = self.u_prev + self.dt * (-u_adv + self.nu * u_diff)
        v_star = self.v_prev + self.dt * (-v_adv + self.nu * v_diff)
        
                                               
        u_star[:, 0] = self.u[:, 0]         
        v_star[:, 0] = 0.0                   
        
        u_star[:, -1] = u_star[:, -2]          
        v_star[:, -1] = v_star[:, -2]          
                              
        u_star[0, :] = 0.0 
        v_star[0, :] = 0.0
        u_star[-1, :] = 0.0
        v_star[-1, :] = 0.0
        
                                
        u_star.mul_(self.mask)
        v_star.mul_(self.mask)
        
        p_new = self.pressure_poisson(u_star, v_star, max_iter=30, tolerance=1e-4)
        
                                               
        if torch.isnan(p_new).any() or torch.isinf(p_new).any():
                                            
            self.dt *= 0.5
            print(f"Reducing time step to {self.dt:.6f} due to pressure instability")
            return False
        
                                                                        
                                      
        dp_dx = torch.zeros_like(p_new)
        dp_dy = torch.zeros_like(p_new)
        
        dp_dx[1:-1, 1:-1] = (p_new[1:-1, 2:] - p_new[1:-1, 0:-2]) / (2.0 * self.dx)
        dp_dy[1:-1, 1:-1] = (p_new[2:, 1:-1] - p_new[0:-2, 1:-1]) / (2.0 * self.dy)
        
                                                      
        u_next = u_star - self.dt * dp_dx
        v_next = v_star - self.dt * dp_dy
        
                                         
        u_next[:, 0] = self.u[:, 0]         
        v_next[:, 0] = 0.0                   
        
        u_next[:, -1] = u_next[:, -2]          
        v_next[:, -1] = v_next[:, -2]          
        
                              
        u_next[0, :] = 0.0
        v_next[0, :] = 0.0
        u_next[-1, :] = 0.0
        v_next[-1, :] = 0.0
        
                                                          
        u_next.mul_(self.mask)
        v_next.mul_(self.mask)
                                          
        if (torch.isnan(u_next).any() or 
            torch.isnan(v_next).any() or
            torch.isinf(u_next).any() or
            torch.isinf(v_next).any()):
                                            
            self.dt *= 0.5
            print(f"Reducing time step to {self.dt:.6f} due to velocity instability")
            return False
        
                                                                               
        self.u.copy_(torch.clamp(u_next, -10.0, 10.0))
        self.v.copy_(torch.clamp(v_next, -10.0, 10.0))
        self.p.copy_(p_new)
        
        return True

    def run_simulation(self, n_steps=1000, check_steady=True, steady_tol=1e-5, verbosity=1):
        safety_factor = 0.25
        grid_factor = min(self.dx, self.dy)**2
        max_vel = max(torch.max(torch.abs(self.u[:, 0])).item(), 1.0)
        suggested_dt = safety_factor * grid_factor / max_vel
        self.dt = min(suggested_dt, 0.001)
    
                                      
        iterator = range(n_steps)
        if verbosity >= 1:
            print(f"Using adaptive time step: {self.dt:.6f}")
            iterator = tqdm(iterator, desc=f"Running NS Solver (Re={self.Re})")
        
                                         
        velocity_change = float('inf')
        failed_steps = 0
        min_dt = self.dt
        max_failed_steps = 20
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            if hasattr(torch.backends.cuda, 'matmul'):
                if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
                    torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True
        
        for step in iterator:
            if check_steady and step % 10 == 0:
                u_old = self.u.clone()
                v_old = self.v.clone()
            
            try:
                time_step_success = self.time_step()
            except RuntimeError as e:
                print(f"Error during time step: {e}")
                self.dt *= 0.5
                print(f"Reducing time step to {self.dt:.6f} due to runtime error")
                time_step_success = False
            
                                       
            if not time_step_success:
                failed_steps += 1
                if verbosity >= 2:
                    print(f"Time step failed, reducing dt to {self.dt:.6f}")
                min_dt = min(min_dt, self.dt)
                continue
            
                                                    
            if step % 50 == 0 and failed_steps == 0 and self.dt < min_dt * 2:
                self.dt *= 1.1
                self.dt = min(self.dt, 0.001)
                if verbosity >= 2:
                    print(f"Increasing dt to {self.dt:.6f}")
            
                                        
            if failed_steps > max_failed_steps:
                if verbosity >= 1:
                    print(f"Too many failed steps ({failed_steps}), ending simulation early")
                break
            
            failed_steps = 0
            
            if check_steady and step % 10 == 0:
                if (torch.isnan(self.u).any() or torch.isnan(self.v).any() or 
                    torch.isinf(self.u).any() or torch.isinf(self.v).any()):
                    if verbosity >= 1:
                        print("Simulation became unstable with NaN/Inf values")
                    break
                
                with torch.no_grad():
                    velocity_change = max(
                        torch.max(torch.abs(self.u - u_old)).item(),
                        torch.max(torch.abs(self.v - v_old)).item()
                    )
                
                if verbosity >= 2:
                    print(f"Step {step}, Velocity change: {velocity_change:.6f}")
                
                if velocity_change < steady_tol:
                    if verbosity >= 1:
                        print(f"Reached steady state after {step} steps")
                    break
        
        u_result = self.u.detach().cpu().numpy()
        v_result = self.v.detach().cpu().numpy()
        p_result = self.p.detach().cpu().numpy()
        mask_result = self.mask.detach().cpu().numpy()
        
                                     
        u_result = np.nan_to_num(u_result, nan=0.0, posinf=0.0, neginf=0.0)
        v_result = np.nan_to_num(v_result, nan=0.0, posinf=0.0, neginf=0.0)
        p_result = np.nan_to_num(p_result, nan=0.0, posinf=0.0, neginf=0.0)
        
        velocity_mag = np.sqrt(u_result**2 + v_result**2)
        
        results = {
            'u': u_result,
            'v': v_result,
            'p': p_result,
            'velocity_mag': velocity_mag,
            'mask': mask_result,
            'Re': self.Re,
            'steps': step + 1,
            'steady': velocity_change < steady_tol if check_steady else False,
            'dt_final': self.dt,
            'failed_steps': failed_steps
        }
        
        return results

    def to_cpu(self):
        self.u = self.u.cpu()
        self.v = self.v.cpu()
        self.p = self.p.cpu()
        self.mask = self.mask.cpu()
        self.fluid_mask = self.fluid_mask.cpu()
        self.u_prev = self.u_prev.cpu()
        self.v_prev = self.v_prev.cpu()
        self.tmp_u = self.tmp_u.cpu()
        self.tmp_v = self.tmp_v.cpu()
        self.tmp_p = self.tmp_p.cpu()
        self.device = 'cpu'
        
                                  
        torch.cuda.empty_cache()
        
    def to_gpu(self, device='cuda'):
        if torch.cuda.is_available():
            self.device = device
            self.u = self.u.to(device)
            self.v = self.v.to(device)
            self.p = self.p.to(device)
            self.mask = self.mask.to(device)
            self.fluid_mask = self.fluid_mask.to(device)
            self.u_prev = self.u_prev.to(device)
            self.v_prev = self.v_prev.to(device)
            self.tmp_u = self.tmp_u.to(device)
            self.tmp_v = self.tmp_v.to(device)
            self.tmp_p = self.tmp_p.to(device)
        else:
            print("CUDA not available, staying on CPU")
