import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from itertools import combinations

# 1. Load data
path_df = pd.read_csv('optimization_path.csv') # [2]
acc_df = pd.read_csv('accuracy_history.csv') # [1]
X = path_df.values
y = acc_df['BO_Acc'].values 
best_idx = 18 # Iteration 19 is the peak accuracy of 0.9668 [1]

# 2. Fit Surrogate Model
gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-6, normalize_y=True)
gp.fit(X, y)

cols = path_df.columns
pairs = list(combinations(range(len(cols)), 2))

for idx1, idx2 in pairs:
    name1, name2 = cols[idx1], cols[idx2]
    plt.figure(figsize=(8, 6))
    
    # Create 2D slice grid
    x1_min, x1_max = X[:, idx1].min(), X[:, idx1].max()
    x2_min, x2_max = X[:, idx2].min(), X[:, idx2].max()
    
    if "Learning_Rate" == name1:
        x1_range = np.logspace(np.log10(x1_min), np.log10(x1_max), 100)
    else:
        x1_range = np.linspace(x1_min, x1_max, 100)
        
    if "Learning_Rate" == name2:
        x2_range = np.logspace(np.log10(x2_min), np.log10(x2_max), 100)
    else:
        x2_range = np.linspace(x2_min, x2_max, 100)

    X1, X2 = np.meshgrid(x1_range, x2_range)
    grid_points = np.tile(X[best_idx], (X1.size, 1))
    grid_points[:, idx1] = X1.ravel()
    grid_points[:, idx2] = X2.ravel()
    
    Z = gp.predict(grid_points).reshape(X1.shape)
    
    # Plot components
    plt.contourf(X1, X2, Z, levels=25, cmap='viridis')
    plt.colorbar(label='Accuracy') # Thermostat for acc
    
    plt.plot(X[:, idx1], X[:, idx2], 'r:', alpha=0.5, label='Path') # Path mapping [3]
    plt.scatter(X[:, idx1], X[:, idx2], c='red', s=30, edgecolors='black')
    plt.scatter(X[best_idx, idx1], X[best_idx, idx2], c='blue', s=100, label='Sweet Spot', zorder=5)
    
    # Iteration numbering [3]
    for i, (px, py) in enumerate(zip(X[:, idx1], X[:, idx2])):
        plt.text(px, py, str(i+1), color='white', fontsize=8, fontweight='bold')

    plt.xlabel(name1)
    plt.ylabel(name2)
    if "Learning_Rate" == name1: plt.xscale('log')
    if "Learning_Rate" == name2: plt.yscale('log')
    
    plt.legend(loc='upper right', fontsize='x-small')
    plt.title(f'Slice: {name1} vs {name2}')
    
    # Save individual file
    filename = f"plot_{name1}_vs_{name2}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {filename}")