# Fourier Neural Operator for Airfoil Flow Simulation

<div align="center">

![FNO Airfoil Flow](results/visualizations/sample_0.png)

</div>

This repository contains an implementation of the Fourier Neural Operator (FNO) for predicting turbulent flow around NACA airfoils at varying Reynolds numbers. The FNO model serves as a surrogate for traditional Computational Fluid Dynamics (CFD) simulations, providing orders of magnitude speedup while maintaining high accuracy.

## Key Features

- **Ultra-Fast Flow Prediction**: Generate flow fields in milliseconds instead of hours with traditional CFD
- **Resolution Invariance**: Train on one grid resolution, predict on others
- **Multi-Resolution Training**: Enhanced generalization through training on varied resolutions
- **Uncertainty Quantification**: Monte Carlo dropout for prediction confidence estimation
- **Physics-Informed Constraints**: Enforce physical properties like the divergence-free condition
- **Interactive Demo**: Web-based interface for real-time airfoil flow prediction

## Project Structure

```
.
├── data/                 # Dataset directory
├── data_generation/      # CFD simulation code for data generation
├── model/                # FNO model implementation
├── training/             # Training scripts and utilities
├── inference/            # Inference and visualization modules
├── evaluation/           # Model evaluation and metrics
├── demo/                 # Interactive web demo application
├── checkpoints/          # Saved model checkpoints
└── results/              # Analysis results and visualizations
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/fourier-neural-operator.git
cd fourier-neural-operator

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

{{ ... }}

# Install dependencies
pip install -r requirements.txt

## Data Generation

Generate training data by running CFD simulations for various Reynolds numbers and airfoil shapes:

```bash
python data_generation/generate_data.py \
    --output_dir ./data \
    --n_samples 200 \
    --re_min 1000 \
    --re_max 10000 \
    --grid_size 128 128 \
    --airfoil_types 0012 2412 4412 6412
```

The data generation process:
1. Creates structured grids around NACA airfoils
2. Solves the incompressible Navier-Stokes equations using a GPU-accelerated solver
3. Collects steady-state velocity and pressure fields
4. Normalizes and splits the data into training, validation, and test sets

## Training

Train the FNO model on the generated dataset:

```bash
python training/train.py \
    --data_dir ./data \
    --checkpoint_dir ./checkpoints \
    --modes 12 \
    --width 32 \
    --n_layers 4 \
    --batch_size 16 \
    --epochs 500 \
    --lr 0.001 \
    --use_amp \
    --scheduler cosine
```

Training features:
- **Multi-resolution Training**: Trains on varying grid resolutions for better generalization
- **Automatic Mixed Precision**: Speeds up training while maintaining accuracy
- **Learning Rate Scheduling**: Cosine annealing to improve convergence
- **Checkpoint Management**: Saves best model based on validation performance
- **Early Stopping**: Prevents overfitting by monitoring validation loss

## Inference and Analysis

Generate predictions and visualizations using a trained model:

```bash
python analyze_fno.py \
    --model ./checkpoints/model_best.pth \
    --data_dir ./data \
    --output_dir ./results
```

The analysis script provides:
- Quantitative evaluation metrics (relative L2 error, divergence error)
- Side-by-side comparisons of predicted and ground truth flow fields
- Visualizations of prediction uncertainty using Monte Carlo dropout
- Performance benchmarks (inference time, memory usage)

## Evaluation

Comprehensively evaluate the model's performance and physical correctness:

```bash
python evaluation/evaluate.py \
    --model_path ./checkpoints/model_best.pth \
    --data_dir ./data \
    --output_dir ./evaluation_results
```

The evaluation covers:
- **Accuracy Metrics**: Relative L2 error, maximum pointwise error
- **Physics Compliance**: Divergence of velocity field (mass conservation)
- **Resolution Invariance**: Performance across different grid resolutions
- **Computational Efficiency**: Inference time comparisons vs. traditional CFD

## Interactive Demo

Run the web-based demo application:

```bash
python demo/app.py --model ./checkpoints/model_best.pth
```

The demo provides:
- Interactive selection of Reynolds number (1,000-100,000)
- NACA airfoil type selection
- Real-time visualization of velocity and pressure fields
- Streamline plotting for flow pattern visualization
- Performance statistics

The application will be available at http://localhost:8000

## Model Performance

| Metric | Value |
|--------|-------|
| Mean Relative L2 Error | 0.0845 ± 0.0312 |
| Mean Divergence Error | 0.000123 |
| Inference Time | 2.85 ms per sample |
| Model Parameters | 1,184,323 |
| GPU Memory Usage | 0.45 GB |

## Technical Details

### Fourier Neural Operator

The Fourier Neural Operator (FNO) learns the mapping between function spaces by parameterizing the integral kernel directly in Fourier space. This allows it to efficiently capture global dependencies and long-range interactions that are essential for fluid dynamics.

Key advantages of FNO for fluid flow prediction:

1. **Resolution Invariance**: Can be trained on one resolution and evaluated on another
2. **Global Receptive Field**: Capture long-range dependencies through spectral convolutions
3. **Parameter Efficiency**: Fewer parameters than comparable CNN or transformer models
4. **Fast Inference**: Forward pass is significantly faster than traditional CFD simulations

### Model Architecture

Our FNO implementation consists of:

- **Spectral Convolution Layers**: Process data in the Fourier domain for global interactions
- **Lifting/Projection Layers**: Transform between input/output and feature spaces
- **Residual Connections**: Improve gradient flow during training
- **Batch Normalization**: Stabilize training and improve convergence
- **Dropout**: Enable uncertainty quantification via Monte Carlo sampling

### Physics-Informed Constraints

Our implementation incorporates physics-informed constraints to ensure physically realistic predictions:

- **Divergence-Free Condition**: Enforce conservation of mass for incompressible flow
- **Boundary Conditions**: Apply no-slip boundary condition at airfoil surfaces
- **Conservation Laws**: Verify adherence to momentum conservation

## References

1. Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2020). Fourier Neural Operator for Parametric Partial Differential Equations. arXiv preprint arXiv:2010.08895.

2. Wang, S., Wang, H., & Perdikaris, P. (2021). Learning the solution operator of parametric partial differential equations with physics-informed DeepONets. Science Advances, 7(40), eabi8605.

3. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686-707.

4. Li et al., *Fourier Neural Operator for Parametric PDEs* ([Papers with Code](https://paperswithcode.com/paper/fourier-neural-operator-for-parametric-1))

5. [`neuraloperator`](https://github.com/neuraloperator/neuraloperator) - Official FNO implementation

6. Ning Liu et al., *Domain Agnostic Fourier Neural Operators* ([arXiv](https://arxiv.org/abs/2305.00478))