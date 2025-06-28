import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon

def naca_4digit(digits, n_points=100, chord=1.0):
    m = int(digits[0]) / 100.0
    p = int(digits[1]) / 10.0
    t = int(digits[2:]) / 100.0
    x = np.linspace(0, chord, n_points)
    x_c = x / chord
    yt = 5 * t * (0.2969 * np.sqrt(x_c) - 0.1260 * x_c - 0.3516 * x_c**2 + 0.2843 * x_c**3 - 0.1015 * x_c**4)
    if p > 0 and m > 0:
        yc = np.where(x_c < p, m * (x_c / p**2) * (2*p - x_c), m * ((1 - x_c) / (1 - p)**2) * (1 + x_c - 2*p))
        dyc_dx = np.where(x_c < p, 2 * m / p**2 * (p - x_c), 2 * m / (1 - p)**2 * (p - x_c))
        
        theta = np.arctan(dyc_dx)
    else:
        yc = np.zeros_like(x_c)
        theta = np.zeros_like(x_c)
    
                                                   
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    x = np.concatenate([np.flip(xl), xu[1:]])
    y = np.concatenate([np.flip(yl), yu[1:]])
    return x, y


def create_airfoil_mask(grid_size, airfoil_coords, padding=0.1):
    height, width = grid_size
    x, y = airfoil_coords
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)
    x_range = x_max - x_min
    y_range = y_max - y_min
    pad_x = padding * width
    pad_y = padding * height
    scale_x = (width - 2 * pad_x) / x_range
    scale_y = (height - 2 * pad_y) / y_range
    scale = min(scale_x, scale_y)
    x_scaled = ((x - x_min) * scale + pad_x).astype(int)
    y_scaled = ((y - y_min) * scale + pad_y).astype(int)
    mask = np.ones((height, width), dtype=np.int8)
    rr, cc = polygon(y_scaled, x_scaled)
    valid_indices = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
    rr, cc = rr[valid_indices], cc[valid_indices]
    mask[rr, cc] = 0
    return mask


def generate_structured_grid(grid_size, airfoil_type='0012'):
    height, width = grid_size
    x, y = naca_4digit(airfoil_type, n_points=200)
    mask = create_airfoil_mask(grid_size, (x, y))
    y_grid, x_grid = np.meshgrid(np.linspace(0, 1, height), np.linspace(0, 1, width), indexing='ij')
    grid = {
        'x': x_grid,
        'y': y_grid,
        'mask': mask,
        'airfoil_coords': (x, y),
        'airfoil_type': airfoil_type
    }
    
    return grid


def visualize_grid(grid, title='Structured Grid with NACA Airfoil'):
    plt.figure(figsize=(12, 10))
    plt.imshow(grid['mask'], cmap='gray', origin='lower')
                                      
    x, y = grid['airfoil_coords']
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)
    height, width = grid['mask'].shape
    pad_x = 0.1 * width
    pad_y = 0.1 * height
    x_range = x_max - x_min
    y_range = y_max - y_min
    scale_x = (width - 2 * pad_x) / x_range
    scale_y = (height - 2 * pad_y) / y_range
    scale = min(scale_x, scale_y)
    x_scaled = (x - x_min) * scale + pad_x
    y_scaled = (y - y_min) * scale + pad_y
    plt.plot(x_scaled, y_scaled, 'r-', linewidth=2)
    plt.title(title)
    plt.tight_layout()
    plt.colorbar(label='Fluid Domain (1) / Solid (0)')
    return plt


if __name__ == "__main__":
    grid_size = (128, 128)
    airfoil_type = '0012'
    grid = generate_structured_grid(grid_size, airfoil_type)
    plt_obj = visualize_grid(grid, f'NACA {airfoil_type} on {grid_size[0]}×{grid_size[1]} Grid')
    plt_obj.savefig(f'naca_{airfoil_type}_grid.png', dpi=150)
    plt_obj.show()
