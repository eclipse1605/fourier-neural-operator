import os
import sys
from generate_data import normalize_dataset

if __name__ == '__main__':
    data_dir = sys.argv[1] if len(sys.argv) > 1 else './data'
    print(f"Running normalization on dataset in: {data_dir}")
    stats = normalize_dataset(data_dir)
    print("Normalization completed successfully!")
    print(f"Statistics saved to: {os.path.join(data_dir, 'normalization_stats.npz')}")
