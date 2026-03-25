import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_opt_path_2d(file_prefix):
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


def plot_noptima(file_prefix):
    fig, ax = plt.subplots(figsize=(5, 5))

    df = pd.read_csv(f"{file_prefix}-noptima.csv")

    df["objective"] = df["objective"].astype(float)
    df["noptimal"] = df["noptimal"].astype(bool)
    df["x_0"] = df["x_0"].astype(float)
    df["x_1"] = df["x_1"].astype(float)
    sns.scatterplot(
        data=df,
        x="x_0",
        y="x_1",
        hue="objective",
        palette="viridis",
        # sizes="obj",
        legend=False,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def plot_vesa(file_prefix):
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


def plot_shannon(file_prefix):
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


# facilitates avoiding duplicate matplotlib import on scripts which import these functions
# i.e. `import plotlogs as p; p.plot(...); p.show()`
show = plt.show

if __name__ == "__main__":
    FILE_PREFIX = "logs/testprob-z"
    plot_noptima(FILE_PREFIX)
    plot_stat_evolution(FILE_PREFIX)
    plot_vesa(FILE_PREFIX)
    plot_shannon(FILE_PREFIX)
    show()
