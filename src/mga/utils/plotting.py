import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_opt_path_2d(file_prefix):
    fig, ax = plt.subplots(figsize=(5, 5))

    df = pd.read_csv(
        file_prefix + "-evolution.csv",
        header=None,
    )
    df.columns = ["iteration", "niche", "obj", "fit", "x", "y"]
    for i, niche in enumerate(df["niche"].unique()):
        ax.plot(
            df.loc[df.niche == niche, "x"],
            df.loc[df.niche == niche, "y"],
            color=f"C{i}",
            label=niche,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()


def plot_noptima(file_prefix):
    fig, ax = plt.subplots(figsize=(5, 5))

    df = pd.read_csv(
        file_prefix + "-noptima.csv",
        header=None,
    )
    df = df.T
    df.columns = ["niche", "obj", "nopt", "x", "y"]
    df["obj"] = df["obj"].astype(float)
    df["nopt"] = df["nopt"].astype(bool)
    df["x"] = df["x"].astype(float)
    df["y"] = df["y"].astype(float)
    sns.scatterplot(
        df,
        x="x",
        y="y",
        hue="obj",
        palette="viridis",
        # sizes="obj",
        legend=False,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def plot_vesa(file_prefix):
    fig, ax = plt.subplots()

    df = pd.read_csv(
        file_prefix + "-diversity.csv",
        header=0,
        usecols=["VESA"],
    )
    df = df.reset_index(names="iteration")

    sns.lineplot(
        df,
        x="iteration",
        y="VESA",
        ax=ax,
        legend=False,
    )

    ax.set_title("VESA")


def plot_shannon(file_prefix):
    fig, ax = plt.subplots()

    df = pd.read_csv(
        file_prefix + "-diversity.csv",
        header=0,
        usecols=["shannon"],
    )
    df = df.reset_index(names="iteration")

    sns.lineplot(
        df,
        x="iteration",
        y="shannon",
        ax=ax,
        legend=False,
    )

    ax.set_title("Shannon")


def plot_stat_evolution(file_prefix):
    fig, axs = plt.subplots(3, 3, sharex=True, layout="tight")
    axs = axs.flatten()
    ax_idx = -1

    ax_idx += 1

    df = pd.read_csv(
        file_prefix + "-nobjective.csv",
        header=None,
    )
    df = df.reset_index()
    niches = ["optimal"] + [f"nopt{n}" for n in range(df.shape[1] - 2)]
    df.columns = ["iteration"] + niches
    df = df.melt(
        id_vars="iteration",
        value_vars=niches,
        var_name="niche",
        value_name="objective",
        ignore_index=True,
    )

    sns.lineplot(
        df,
        x="iteration",
        y="objective",
        hue="niche",
        hue_order=["reserved"] + niches,
        ax=axs[ax_idx],
        legend=False,
    )

    axs[ax_idx].set_title("objective")

    ax_idx += 1

    df = pd.read_csv(
        file_prefix + "-nfitness.csv",
        header=None,
    )
    df = df.reset_index()
    niches = ["optimal"] + [f"nopt{n}" for n in range(df.shape[1] - 2)]
    df.columns = ["iteration"] + niches
    df = df.melt(
        id_vars="iteration",
        value_vars=niches,
        var_name="niche",
        value_name="fitness",
        ignore_index=True,
    )

    sns.lineplot(
        df,
        x="iteration",
        y="fitness",
        hue="niche",
        hue_order=["reserved"] + niches,
        ax=axs[ax_idx],
        legend=False,
    )

    axs[ax_idx].set_title("fitness")

    ax_idx += 1

    mean_of_fitness = df.groupby("iteration")["fitness"].mean().reset_index()

    sns.lineplot(
        mean_of_fitness,
        x="iteration",
        y="fitness",
        ax=axs[ax_idx],
        legend=False,
    )

    axs[ax_idx].set_title("mean of fitness " + str(round(100 * mean_of_fitness.fitness.iloc[-1], 2)))
    for metric in ("std", "var"):
        for agg in ("min", "mean", "max"):
            ax_idx += 1
            name = f"{agg}_{metric}_fit"

            df = pd.read_csv(file_prefix + "-diversity.csv", header=0, usecols=[name])
            df = df.reset_index(names="iteration")

            sns.lineplot(
                df,
                x="iteration",
                y=name,
                ax=axs[ax_idx],
                legend=False,
            )

            axs[ax_idx].set_title(name + " " + str(round(100 * df[name].iloc[-1], 2)))
    # axs[-1].set_xticks(range(0, 201, 25))


# facilitates avoiding duplicate matplotlib import on scripts which import these functions
# i.e. `import plotlogs as p; p.plot(...); p.show()``
show = plt.show

if __name__ == "__main__":
    FILE_PREFIX = "logs/testprob-z"
    plot_noptima(FILE_PREFIX)
    plot_stat_evolution(FILE_PREFIX)
    plot_vesa(FILE_PREFIX)
    plot_shannon(FILE_PREFIX)
    show()
