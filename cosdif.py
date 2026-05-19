import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd  # noqa: F401
import pyvista as pv

os.environ["MGA_JIT_ENABLED"] = "0"

from mga.commons.numba_overload import njit  # noqa: E402


@njit(fastmath=True)
def convex_objective(values_array):
    """
    2D trig function with a single local minimum.
    """
    z = np.full(values_array.shape[0], 3.15, dtype=np.float64)
    for i in range(values_array.shape[0]):
        for j in range(values_array.shape[1]):
            z[i] += np.cos(3 * (values_array[i, j] - 0.5))
    return z


def plot_cosine_difference(v1, v2, w_vals, origin=None):
    v1, v2 = np.asarray(v1, dtype=float), np.asarray(v2, dtype=float)
    if origin is None:
        origin = np.zeros(2)

    assert len(v1) == len(v2) == len(origin) == 2

    origin = np.append(origin, convex_objective(np.atleast_2d(origin)))
    v1 = np.append(v1, convex_objective(np.atleast_2d(v1)))
    v2 = np.append(v2, convex_objective(np.atleast_2d(v2)))

    # Broadcast base vectors and scale the 3rd dimension (index 2)
    v1_w = np.broadcast_to(v1, (len(w_vals), 3)).copy()
    v2_w = np.broadcast_to(v2, (len(w_vals), 3)).copy()

    v1_w -= origin
    v2_w -= origin

    v1_w[:, 2] *= w_vals
    v2_w[:, 2] *= w_vals

    # Vectorized dot product and norms
    dot_prod = np.einsum('ij,ij->i', v1_w, v2_w)
    norm1 = np.linalg.norm(v1_w, axis=1)
    norm2 = np.linalg.norm(v2_w, axis=1)

    # Calculate difference, handling origin zero-vectors
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_sim = dot_prod / (norm1 * norm2)
        cos_sim = np.nan_to_num(cos_sim, nan=1.0)  # Cosine similarity of 0-vector is conventionally 1 (0 diff)

    cos_diff = 1 - cos_sim

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(w_vals, cos_diff, color='black', linewidth=1.5)
    plt.xlabel(r"Weight")
    plt.xscale("log")
    plt.ylabel("Cosine Difference")
    plt.title(f"Cosine Difference vs. Weight\nv1={v1}\nv2={v2}")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def plot_cosine_diff_surface(v1_2d, w_vals, s_vals, origin=None, v2_2d=None, locus="linear"):
    v1_2d = np.asarray(v1_2d, dtype=float)
    if origin is None:
        origin = np.zeros(2)
    else:
        origin = np.asarray(origin, dtype=float)

    if v2_2d is None:
        v2_2d = v1_2d
    else:
        v2_2d = np.asarray(v2_2d, dtype=float)

    assert len(v1_2d) == len(origin) == len(v2_2d) == 2

    # Calculate 3D points for origin and base v1
    z_origin = convex_objective(np.atleast_2d(origin))[0]
    O_3d = np.array([origin[0], origin[1], z_origin])

    z1 = convex_objective(np.atleast_2d(v1_2d))[0]
    v1_3d = np.array([v1_2d[0], v1_2d[1], z1])

    # Create grid for W (weights) and S (scales/rotations)
    W, S = np.meshgrid(w_vals, s_vals, indexing='ij')
    W_flat = W.flatten()
    S_flat = S.flatten()

    dir_2d = v2_2d - origin
    
    # Calculate V2 2D coordinates and trajectory for plot based on locus
    if locus == "linear":
        v2_2d_flat = origin + np.outer(S_flat, dir_2d)
        
        v2_min = origin + np.min(s_vals) * dir_2d
        v2_max = origin + np.max(s_vals) * dir_2d
        traj_x = [v2_min[0], v2_max[0]]
        traj_y = [v2_min[1], v2_max[1]]
        s_label = "Scale Factor (s)"
        
    elif locus == "circular":
        R = np.linalg.norm(dir_2d)
        theta_0 = np.arctan2(dir_2d[1], dir_2d[0])
        theta_s = theta_0 + S_flat * np.pi
        v2_2d_flat = origin + np.column_stack((R * np.cos(theta_s), R * np.sin(theta_s)))
        
        # Calculate arc for the 2D plot
        theta_arc = theta_0 + np.linspace(np.min(s_vals), np.max(s_vals), 100) * np.pi
        traj_x = origin[0] + R * np.cos(theta_arc)
        traj_y = origin[1] + R * np.sin(theta_arc)
        s_label = "Rotation Phase (s)"
        
    else:
        raise ValueError("locus must be 'linear' or 'circular'")

    # --- 2D Trajectory Visualization Snippet ---
    plt.figure(figsize=(6, 6))

    plt.plot(traj_x, traj_y, color='gray', linestyle='--', label=f'V2 Trajectory ({locus})')

    plt.scatter(origin[0], origin[1], color='red', zorder=5, label='Origin')
    plt.scatter(v1_2d[0], v1_2d[1], color='blue', zorder=5, label='V1 Base')
    
    if not np.array_equal(v1_2d, v2_2d):
        plt.scatter(v2_2d[0], v2_2d[1], color='green', zorder=5, label='V2 Base')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'2D Projection of Vectors ({locus.capitalize()} Locus)')
    plt.axhline(0, color='black', linewidth=0.5, alpha=0.3)
    plt.axvline(0, color='black', linewidth=0.5, alpha=0.3)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.axis('equal')  
    plt.show()
    # -------------------------------------------

    # Calculate V2 Z-coordinates and stack to 3D
    z2_flat = convex_objective(v2_2d_flat)
    v2_3d_flat = np.column_stack((v2_2d_flat, z2_flat))

    # Apply 3D origin shift
    V1_shifted = v1_3d - O_3d
    V2_shifted = v2_3d_flat - O_3d

    # Apply W scaling to the Z dimension
    V1_w = np.broadcast_to(V1_shifted, (len(W_flat), 3)).copy()
    V1_w[:, 2] *= W_flat

    V2_w = V2_shifted.copy()
    V2_w[:, 2] *= W_flat

    # Vectorized cosine difference
    dot_prod = np.einsum('ij,ij->i', V1_w, V2_w)
    norm1 = np.linalg.norm(V1_w, axis=1)
    norm2 = np.linalg.norm(V2_w, axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        cos_sim = dot_prod / (norm1 * norm2)
        cos_sim = np.nan_to_num(cos_sim, nan=1.0)

    cos_diff = 1 - cos_sim
    Z_grid = cos_diff.reshape(W.shape)

    # PyVista Structured Grid Plotting
    X_grid = np.log10(W) if np.all(W > 0) else W
    x_label = "log10(Weight)" if np.all(W > 0) else "Weight"

    grid = pv.StructuredGrid(X_grid, S, Z_grid)
    grid["Cosine Difference"] = Z_grid.flatten(order="F")

    plotter = pv.Plotter()
    plotter.add_mesh(grid, scalars="Cosine Difference", cmap="viridis", show_edges=True)

    plotter.show_bounds(
        grid='back',
        location='outer',
        xlabel=x_label,
        ylabel=s_label,
        zlabel="Cosine Difference"
    )
    plotter.add_title(f"Cosine Difference vs. Weight and Scale ({locus.capitalize()} Locus)")
    plotter.show()


def plot_cosine_diff_heatmap(v1_2d, x_bounds=(-1, 2), y_bounds=(-1, 2), resolution=100, origin=None, w=1.0, z_contour=None):
    v1_2d = np.asarray(v1_2d, dtype=float)
    if origin is None:
        origin = np.zeros(2)
    else:
        origin = np.asarray(origin, dtype=float)

    assert len(v1_2d) == len(origin) == 2
     # Calculate 3D points for origin and v1
    z_origin = convex_objective(np.atleast_2d(origin))[0]
    O_3d = np.array([origin[0], origin[1], z_origin])

    z1 = convex_objective(np.atleast_2d(v1_2d))[0]
    v1_3d = np.array([v1_2d[0], v1_2d[1], z1])

    # Create 2D grid for V2 candidates
    x_vals = np.linspace(x_bounds[0], x_bounds[1], resolution)
    y_vals = np.linspace(y_bounds[0], y_bounds[1], resolution)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    v2_2d_flat = np.column_stack((X.ravel(), Y.ravel()))

    # Calculate Z for all V2 points
    z2_flat = convex_objective(v2_2d_flat)
    v2_3d_flat = np.column_stack((v2_2d_flat, z2_flat))

    # Apply 3D origin shift
    V1_shifted = v1_3d - O_3d
    V2_shifted = v2_3d_flat - O_3d

    # Apply W scaling to the Z dimension
    V1_shifted[2] *= w
    V2_shifted[:, 2] *= w

    # Vectorized cosine difference (broadcasting V1 against all V2s)
    dot_prod = np.sum(V2_shifted * V1_shifted, axis=1)
    norm1 = np.linalg.norm(V1_shifted)
    norm2 = np.linalg.norm(V2_shifted, axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        cos_sim = dot_prod / (norm1 * norm2)
        cos_sim = np.nan_to_num(cos_sim, nan=1.0)

    cos_diff = 1 - cos_sim
    diff_grid = cos_diff.reshape(X.shape)
    
    # Reshape objective Z for contour plotting
    Z_obj_grid = z2_flat.reshape(X.shape)

    # Plotting
    plt.figure(figsize=(8, 6))
    
    # Render heatmap
    mesh = plt.pcolormesh(X, Y, diff_grid, cmap='viridis', shading='auto')
    cbar = plt.colorbar(mesh)
    cbar.set_label(f"Cosine Difference (w={w})")

    # Plot objective contour if specified
    if z_contour is not None:
        plt.contour(X, Y, Z_obj_grid, levels=[z_contour], colors='red', linestyles='dotted')
        # Dummy plot for legend
        plt.plot([], [], color='red', linestyle='dotted', label=f'Objective Contour (z={z_contour})')

    # Plot markers
    plt.scatter(origin[0], origin[1], color='red', marker='X', s=100, edgecolor='black', zorder=5, label='Origin')
    plt.scatter(v1_2d[0], v1_2d[1], color='cyan', marker='o', s=80, edgecolor='black', zorder=5, label='V1 Base')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'2D Spatial Heatmap of 3D Cosine Difference, {w=}')
    plt.axhline(origin[1], color='white', linewidth=0.5, alpha=0.5, linestyle='--')
    plt.axvline(origin[0], color='white', linewidth=0.5, alpha=0.5, linestyle='--')
    plt.legend(loc='upper right')
    plt.axis('equal') 
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # origin = np.array([0.5, 0.5])
    # A = np.array([0.5005, 0.1091])
    # B = np.array([0.5, 0.1])
    # w_array = np.logspace(-3, 4, 200)

    # plot_cosine_difference(A, B, w_array, origin)

    origin_pt = np.array([0.5, 0.5])
    v1_pt = np.array([0.25, 0.25])
    v2_pt = np.array([0.25, 0.5])

    w_array = np.logspace(-2, 3, 100)

    # Example 1: Linear
    s_array_linear = np.linspace(0.1, 2.0, 50)
    # plot_cosine_diff_surface(v1_pt, w_array, s_array_linear, origin=origin_pt, locus="linear")
    
    # Example 2: Circular
    s_array_circular = np.linspace(-1, 1, 50) 
    # plot_cosine_diff_surface(
    #     v1_pt,
    #     w_array,
    #     s_array_circular,
    #     origin=origin_pt,
    #     locus="circular",
    #     v2_2d = v2_pt
    # )

    plot_cosine_diff_heatmap(
        v1_pt,
        x_bounds=(0, 1.0),
        y_bounds=(0, 1.0),
        resolution=200,
        origin=origin_pt,
        w=0.1,
        z_contour=(1-0.12)*5.15
    )

