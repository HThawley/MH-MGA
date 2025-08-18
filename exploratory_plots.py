# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 14:30:50 2025

@author: hmtha
"""

import matplotlib.pyplot as plt 
import matplotlib.colors as plc
import numpy as np


x_min, x_max = 0, 1
y_min, y_max = 0, 1
resolution = 200
cmap='viridis'
#%%
centroids = np.array([
    [0.974, 0.974],
    [0.974, 0.02],
    [0.02, 0.974],
    [0.5, 0.9], 
    [0.9, 0.5],
#     [0.7, 0.25],
#     [0.25, 0.7],
#     [0.75, 0.75],
    
#     # [0.4, 0.5],
    ]
    )

# centroids = np.array([
#     [0.97378074, 0.9738754],
#     [0.54538798, 0.75552233],
#     [0.97515078, 0.02088975],
#     [0.97263192, 0.53812105],
#     [0.02076667, 0.97464276], 
#     ]
#     )


#%%

def min_of_euclidean_distances(p1, ps, degree):
    return min([euclideanDistance(p1, p)**degree for p in ps])

def mean_of_euclidean_distances(p1, ps, degree):
    return np.mean([euclideanDistance(p1, p)**degree for p in ps])

def euclideanDistance(p1, p2):
    """Euclidean distance"""
    return sum((p1-p2)**2)**0.5




fig, axs = plt.subplots(1, 3, figsize=(12,3))

for i in range(3):
    ax = axs.flatten()[i]
    
    # i = 0 
    # fig, ax = plt.subplots()
    
    degree=1/(i+1)
    title=str(degree)
    
    # Create a grid of points
    x_vals = np.linspace(x_min, x_max, resolution)
    y_vals = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Evaluate the function at each point in the grid
    Z = np.zeros((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            point = np.array([X[i, j], Y[i, j]])
            # Z[i, j] = mean_of_euclidean_distances(point, centroids, degree)
            Z[i, j] = mean_of_euclidean_distances(point, centroids, degree)

    log_offset = Z.min()+0.1
    Z = Z + log_offset # Add offset to make all values positive
    # Ensure vmin is still strictly positive after offset
    vmin_log = np.max([Z.min(), log_offset])
    vmin_log = max(vmin_log, 1e-9) # Ensure vmin is not too small to cause rendering issues

    vmax_log = Z.max()
    
    norm = plc.LogNorm(vmin=vmin_log, vmax=vmax_log)

    # Plot the heatmap
    img=ax.imshow(Z, origin='lower', extent=[x_min, x_max, y_min, y_max],
               cmap=cmap, aspect='auto', norm=norm)
    cbar = plt.colorbar(img) # Pass the image object to colorbar
    cbar.set_label('Function Score')
    # Add labels and title
    ax.grid(False) # Heatmaps usually don't need a grid on top of the colors
    ax.set_title(title)

    ax.scatter(centroids[:, 0], centroids[:,1], color="black", zorder=1000, s=100, marker='x')
    
    
plt.show()