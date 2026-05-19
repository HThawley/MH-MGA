import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_opt_path_2d(file_prefix, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    df = pd.read_csv(f"{file_prefix}-evolution.csv")

    for i, niche in enumerate(df["niche_id"].unique()):
        ax.plot(
            df.loc[df.niche_id == niche, "x_0"],
            df.loc[df.niche_id == niche, "x_1"],
            color=f"C{i}",
            label=niche,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()


def plot_noptima(file_prefix, ax=None):
    hue = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
        hue = "objective"

    df = pd.read_csv(f"{file_prefix}-noptima.csv")

    df["objective"] = df["objective"].astype(float)
    df["noptimal"] = df["noptimal"].astype(bool)
    df["x_0"] = df["x_0"].astype(float)
    df["x_1"] = df["x_1"].astype(float)
    sns.scatterplot(
        data=df,
        x="x_0",
        y="x_1",
        hue=hue,
        palette="viridis",
        legend=False,
        ax=ax,
    )

    for i in range(1, len(df)):
        ax.plot(
            [df.loc[0, "x_0"], df.loc[i, "x_0"]],
            [df.loc[0, "x_1"], df.loc[i, "x_1"]],
            linewidth=0.5,
            color="k",
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def plot_vesa(file_prefix, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    df = pd.read_csv(
        f"{file_prefix}-diversity.csv",
        usecols=["iter", "VESA"],
    )

    sns.lineplot(
        data=df,
        x="iter",
        y="VESA",
        ax=ax,
        legend=False,
    )

    ax.set_title("VESA")


def plot_shannon(file_prefix, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    df = pd.read_csv(
        f"{file_prefix}-diversity.csv",
        usecols=["iter", "shannon"],
    )

    sns.lineplot(
        data=df,
        x="iter",
        y="shannon",
        ax=ax,
        legend=False,
    )

    ax.set_title("Shannon")


def plot_stat_evolution(file_prefix):
    fig, axs = plt.subplots(3, 3, sharex=True, layout="tight")
    axs = axs.flatten()

    df = pd.read_csv(f"{file_prefix}-niche_metrics.csv")

    ax_idx = 0
    sns.lineplot(
        data=df,
        x="iter",
        y="objective",
        hue="niche_id",
        ax=axs[ax_idx],
        legend=False,
    )
    axs[ax_idx].set_title("objective")

    ax_idx += 1
    sns.lineplot(
        data=df,
        x="iter",
        y="fitness",
        hue="niche_id",
        ax=axs[ax_idx],
        legend=False,
    )
    axs[ax_idx].set_title("fitness")

    mean_of_fitness = df.groupby("iter")["fitness"].mean().reset_index()

    ax_idx += 1
    sns.lineplot(
        data=mean_of_fitness,
        x="iter",
        y="fitness",
        ax=axs[ax_idx],
        legend=False,
    )

    axs[ax_idx].set_title(f"mean of fitness {100 * mean_of_fitness.fitness.iloc[-1]:.2f}")

    df_div = pd.read_csv(f"{file_prefix}-diversity.csv")

    for metric in ("std", "var"):
        for agg in ("min", "mean", "max"):
            ax_idx += 1
            name = f"{agg}_{metric}_fit"
            sns.lineplot(
                data=df_div,
                x="iter",
                y=name,
                ax=axs[ax_idx],
                legend=False,
            )

            axs[ax_idx].set_title(f"{name} {100 * df_div[name].iloc[-1]:.2f}")
    # axs[-1].set_xticks(range(0, 201, 25))


def _landscape(obj_func, lb, ub, isoline_ratio, ax=None, vmin=None, vmax=None, cbar=False):
    """Plots the 2D landscape of the objective function."""
    if ax is None:
        ax = plt.gca()

    # Generate grid
    x = np.linspace(lb[0], ub[0], 200)
    y = np.linspace(lb[1], ub[1], 200)
    X, Y = np.meshgrid(x, y)

    # Flatten for the njit function
    points = np.column_stack((X.ravel(), Y.ravel()))
    Z = obj_func(points).reshape(X.shape)

    # Plot landscape
    mesh = ax.contourf(X, Y, Z, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)

    # Plot isoline at isoline_ratio * maximum value
    local_max = np.max(Z)
    ax.contour(X, Y, Z, levels=[isoline_ratio * local_max], colors='red', linestyles='dotted')

    if lb[0] == lb[1] and ub[0] == ub[1]:
        ax.set_aspect('equal', adjustable='box')

    if cbar:
        plt.colorbar(mesh, ax=ax)

    return ax, mesh


def plot_landscape(func_1, func_2, lb, ub, isoline_ratio, name_1=None, name_2=None):
    """
    Plots two objective functions side-by-side.
    """
    # Pre-calculate global bounds to scale the shared colorbar correctly
    x = np.linspace(lb[0], ub[0], 200)
    y = np.linspace(lb[1], ub[1], 200)
    X, Y = np.meshgrid(x, y)
    points = np.column_stack((X.ravel(), Y.ravel()))

    z1 = func_1(points)
    z2 = func_2(points)
    global_min = min(z1.min(), z2.min())
    global_max = max(z1.max(), z2.max())

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    ax1, ax2 = axs

    _landscape(func_1, lb, ub, isoline_ratio, ax=ax1, vmin=global_min, vmax=global_max)
    _, mesh = _landscape(func_2, lb, ub, isoline_ratio, ax=ax2, vmin=global_min, vmax=global_max)

    ax1.set_title(name_1)
    ax2.set_title(name_2)

    # Single shared colorbar
    fig.colorbar(mesh, ax=[ax1, ax2], location='right', label='Objective Value')
    return axs


# facilitates avoiding duplicate matplotlib import on scripts which import these functions
# i.e. `import plotlogs as p; p.plot(...); p.show()`
show = plt.show

if __name__ == "__main__":
    FILE_PREFIX = "logs/testprob"
    plot_noptima(FILE_PREFIX)
    plot_stat_evolution(FILE_PREFIX)
    plot_vesa(FILE_PREFIX)
    plot_shannon(FILE_PREFIX)
    show()
