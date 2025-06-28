import os
import argparse
import numpy as np
import torch
import time
import uvicorn
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import io
import base64

                              
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.fno import FNO2d
from data_generation.airfoil import generate_structured_grid, naca_4digit


                    
app = FastAPI(title="FNO Airfoil Flow Predictor", 
              description="Interactive demo for predicting flow around airfoils using Fourier Neural Operator")

                     
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],                     
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

                  
model = None
device = None
normalization_stats = None
grid_size = (128, 128)


@app.on_event("startup")
async def startup_event():
    """Load model and normalization stats at startup with GPU optimization"""
    global model, device, normalization_stats
    
                                       
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name(device)}")
        print(f"CUDA capability: {torch.cuda.get_device_capability(device)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
        
                                                         
        torch.backends.cudnn.benchmark = True
        print("CUDA acceleration enabled for inference")
    else:
        device = torch.device('cpu')
        print("CUDA not available. Using CPU for inference.")
    
                                   
    stats_path = os.path.join(os.path.dirname(__file__), '../data/normalization_stats.npz')
    if os.path.exists(stats_path):
        normalization_stats = dict(np.load(stats_path))
    else:
        print(f"Warning: Normalization stats file not found at {stats_path}")
                                       
        normalization_stats = {
            'u_mean': 0.0, 'u_std': 1.0,
            'v_mean': 0.0, 'v_std': 1.0,
            'p_mean': 0.0, 'p_std': 1.0
        }
    
                
    model_path = os.path.join(os.path.dirname(__file__), '../checkpoints/model_best.pth')
    if os.path.exists(model_path):
                                            
        model = FNO2d(
            in_channels=2,                            
            out_channels=3,                      
            width=32,
            modes1=12,
            modes2=12,
            n_layers=4,
            device=device                                             
        )
        
                      
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
                                                                                     
        print("Using standard model without TorchScript optimization")
        
        print(f"Model loaded from {model_path} with {model.count_params():,} parameters")
    else:
        print(f"Warning: Model file not found at {model_path}")
        print("Running in demo mode without a trained model.")


@app.get("/")
async def get_index():
    return FileResponse(os.path.join(os.path.dirname(__file__), 'static/index.html'))


@app.get("/api/predict")
async def predict(
    re: float = Query(..., ge=1000, le=100000),
    airfoil: str = Query("0012"),
    output_type: str = Query("velocity"),
    use_amp: bool = Query(False)
):
    try:
                                  
        if model is None:
            return await generate_dummy_prediction(re, airfoil, output_type)
        start_time = time.time()
        grid = generate_structured_grid(grid_size, airfoil_type=airfoil)
        mask = grid['mask'].astype(np.float32)                       
        re_normalized = (np.log10(float(re)) - 3) / 2
        re_channel = np.ones(grid_size, dtype=np.float32) * re_normalized
        inputs = np.stack([re_channel, mask], axis=0)
        inputs_tensor = torch.from_numpy(inputs).float().unsqueeze(0).to(device)                       
        with torch.no_grad():
            if use_amp and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    outputs = model(inputs_tensor)
            else:
                outputs = model(inputs_tensor)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        prediction = outputs.detach().cpu().numpy()[0]                                               
        
                             
        u = prediction[0] * normalization_stats['u_std'] + normalization_stats['u_mean']
        v = prediction[1] * normalization_stats['v_std'] + normalization_stats['v_mean']
        p = prediction[2] * normalization_stats['p_std'] + normalization_stats['p_mean']
        u = u * mask
        v = v * mask
        p = p * mask
        velocity_mag = np.sqrt(u**2 + v**2)
        img_base64 = generate_visualization(u, v, p, velocity_mag, mask, re, airfoil, output_type)
        inference_time = time.time() - start_time
        response = {
            "reynolds": re,
            "airfoil": airfoil,
            "visualization": img_base64,
            "max_velocity": float(np.max(velocity_mag)),
            "min_pressure": float(np.min(p)),
            "max_pressure": float(np.max(p)),
            "inference_time_ms": float(inference_time * 1000),
            "gpu_accelerated": torch.cuda.is_available()
        }
        
        return JSONResponse(content=response)
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"ERROR in predict function: {str(e)}\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}. See server logs for details.")


async def generate_dummy_prediction(re, airfoil, output_type):                                          
    grid = generate_structured_grid(grid_size, airfoil_type=airfoil)
    mask = grid['mask']
    x = np.linspace(0, 1, grid_size[1])
    y = np.linspace(0, 1, grid_size[0])
    X, Y = np.meshgrid(x, y)
    scale = re / 50000
                                                           
    u = scale * np.sin(5 * Y) * mask
    v = scale * np.cos(8 * X) * 0.2 * mask
    
                                                   
    wake = np.exp(-50 * (Y - 0.5)**2) * (X > 0.5) * 0.5
    u -= wake * scale * mask
    
    p = scale * (Y - 0.5) * np.exp(-5 * ((X - 0.3)**2 + (Y - 0.5)**2)) * mask
    
    velocity_mag = np.sqrt(u**2 + v**2)
    
    img_base64 = generate_visualization(u, v, p, velocity_mag, mask, re, airfoil, output_type)
    response = {
        "reynolds": re,
        "airfoil": airfoil,
        "visualization": img_base64,
        "max_velocity": float(np.max(velocity_mag)),
        "min_pressure": float(np.min(p)),
        "max_pressure": float(np.max(p)),
        "demo_mode": True
    }
    
    return JSONResponse(content=response)


def generate_visualization(u, v, p, velocity_mag, mask, re, airfoil, output_type):
    plt.figure(figsize=(8, 6))
    
                                       
    if output_type == 'velocity':
        field = velocity_mag
        cmap = 'viridis'
        title = f"Velocity Magnitude, NACA {airfoil}, Re={int(re)}"
    elif output_type == 'pressure':
        field = p
        cmap = 'coolwarm'
        title = f"Pressure Field, NACA {airfoil}, Re={int(re)}"
    elif output_type == 'u_velocity':
        field = u
        cmap = 'RdBu_r'
        title = f"U-Velocity, NACA {airfoil}, Re={int(re)}"
    elif output_type == 'v_velocity':
        field = v
        cmap = 'RdBu_r'
        title = f"V-Velocity, NACA {airfoil}, Re={int(re)}"
    else:
        field = velocity_mag
        cmap = 'viridis'
        title = f"Velocity Magnitude, NACA {airfoil}, Re={int(re)}"
    
    field = np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0)
    
    if output_type in ['u_velocity', 'v_velocity', 'pressure']:
                                                                        
                                              
        abs_field = np.abs(field)
        vmax = np.percentile(abs_field[abs_field > 0], 99) if np.any(abs_field > 0) else 1e-6
        vmax = max(vmax, 1e-6)                   
        im = plt.imshow(field, cmap=cmap, origin='lower', vmin=-vmax, vmax=vmax)
    else:
                                                                           
                                              
        vmax = np.percentile(field[field > 0], 99) if np.any(field > 0) else 1.0
        vmax = max(vmax, 1e-6)                   
        im = plt.imshow(field, cmap=cmap, origin='lower', vmin=0, vmax=vmax)
    
    plt.contour(mask, levels=[0.5], colors='k', linewidths=1.0)
    plt.colorbar(im, label=output_type.replace('_', ' ').title())
    plt.title(title)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plt.close()
    
                                  
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64


@app.get("/api/airfoils")
async def get_airfoil_options():                                  
    airfoils = [
        {"id": "0012", "name": "NACA 0012", "description": "Symmetric 12% thickness"},
        {"id": "2412", "name": "NACA 2412", "description": "2% camber, 12% thickness"},
        {"id": "4412", "name": "NACA 4412", "description": "4% camber, 12% thickness"},
        {"id": "6409", "name": "NACA 6409", "description": "6% camber, 9% thickness"},
        {"id": "0009", "name": "NACA 0009", "description": "Symmetric 9% thickness"},
        {"id": "0015", "name": "NACA 0015", "description": "Symmetric 15% thickness"}
    ]
    return {"airfoils": airfoils}

app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FNO Demo Server with GPU Acceleration")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--enable-gpu", action="store_true", help="Enable GPU acceleration")
    parser.add_argument("--jit-optimize", action="store_true", help="Enable TorchScript JIT optimization")
    parser.add_argument("--profile", action="store_true", help="Enable performance profiling")
    args = parser.parse_args()
    
                                                            
    if args.enable_gpu and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"                 
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"                               
        print("GPU acceleration enabled for server")
    
    print(f"Starting FNO Demo Server at http://{args.host}:{args.port}")
    uvicorn.run(
        "app:app", 
        host=args.host, 
        port=args.port, 
        reload=(not args.profile),                                    
        workers=args.workers,
        log_level="info"
    )
