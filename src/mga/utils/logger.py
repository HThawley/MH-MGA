import numpy as np
from pathlib import Path
from csv import writer
from shutil import copyfile
from os import remove, mkdir
from os.path import exists
from collections.abc import Collection

from mga.metrics import diversity


class FilePrinter:
    """
    Handles writing data to a file with buffering and safe-saving.
    It writes to a temporary file first and then replaces the original
    to prevent corruption during interruption.
    """

    def __init__(
        self,
        file_name: str,
        save_freq: int,
        header: Collection[str] = None,
        resume: bool = False,
        create_dir: bool = True,
    ):
        """
        Initializes the FilePrinter.
        """
        self.file_name = file_name
        self.temp_file_path = "-temp.".join(self.file_name.split("."))
        self.save_freq = save_freq
        self.call_count = 0
        self.buffer = []

        if not resume or not not exists(self.file_name):
            self._create_file(header, create_dir)

    def __call__(self, data_rows: list[list]):
        """
        Adds data to the buffer and writes it out if the save frequency is met.
        """
        self.call_count += 1
        self.buffer.extend(data_rows)

        if (self.save_freq > 0) and (self.call_count % self.save_freq == 0):
            self._flush()

    def _print(self):
        """
        Appends the buffered data to the temporary file.
        """
        with open(self.temp_file_path, "a", newline="") as f:
            csv_writer = writer(f)
            csv_writer.writerows(self.buffer)

    def _copy_and_replace(self):
        """
        Copies the main file to a temp file, appends, and copies back.
        """
        try:
            if exists(self.file_name):
                copyfile(self.file_name, self.temp_file_path)
            self._print()
            copyfile(self.temp_file_path, self.file_name)
        finally:
            if exists(self.temp_file_path):
                remove(self.temp_file_path)

    def _flush(self):
        """
        Writes any buffered data to the file.
        """
        if self.buffer:
            print("\rWriting logs to disk... Do not interrupt.", end="")
            self._copy_and_replace()
            print("\r" + " " * 50, end="\r")
            self.buffer = []

    def _create_file(self, header: Collection[str], create_dir: bool):
        """
        Creates a new, empty log file with an optional header.
        """
        p = Path(self.file_name)
        if create_dir and not exists(p.parent):
            mkdir(p.parent)

        with open(self.file_name, "w", newline="") as f:
            if header is not None:
                writer(f).writerow(header)


class Logger:
    """
    Manages all logging activities for the MGA run.
    It uses multiple FilePrinter instances to log various metrics.
    """

    def __init__(
        self,
        file_prefix: str,
        save_freq: int,
        resume: bool = False,
        detailed: bool = True,
        ndim: int = 0,
        create_dir: bool = True,
    ):
        """
        Initializes all necessary FilePrinters for logging.
        """
        self.detailed = detailed

        self.niche_metrics_printer = FilePrinter(
            file_name=f"{file_prefix}-niche_metric.csv",
            save_freq=save_freq,
            header=["iter", "niche_id", "objective", "penalties", "noptimal", "fitness"],
            resume=resume,
            create_dir=create_dir,
        )

        self.diversity_printer = FilePrinter(
            file_name=f"{file_prefix}-diversity.csv",
            save_freq=save_freq,
            header=[
                "iter",
                "VESA",
                "shannon",
                "mean_std_fit",
                "min_std_fit",
                "max_std_fit",
                "mean_var_fit",
                "min_var_fit",
                "max_var_fit",
                "sum_fit",
                "mean_fit",
            ],
            resume=resume,
        )

        if self.detailed:
            header = ["iter", "niche_id", "objective", "penalties", "noptimal", "fitness"]
            header += [f"x_{i}" for i in range(ndim)] if ndim > 0 else ["..."]

            self.evolution_printer = FilePrinter(
                file_name=f"{file_prefix}-evolution.csv",
                save_freq=save_freq,
                header=header,
                resume=resume,
            )

        self.noptima_printer = FilePrinter(
            file_name=f"{file_prefix}-noptima.csv",
            save_freq=-1,  # Only writes once at the end
            header=[
                "niche_id", "objective", "penalties", "noptimal", "fitness"
            ] + [f"x_{i}" for i in range(ndim)],
            resume=False,  # Always overwrite final results
        )

    def log_iteration(self, iteration: int, population):
        """
        Logs all relevant data for the current iteration.

        Args:
            iteration (int): The current iteration number.
            population (Population): The population object containing current state.
        """
        metrics_data = [
            [
                iteration,
                f"nopt{i}" if i > 0 else "optimum",
                population.optima_raw_objectives[i],
                population.optima_violations[i] * population.violation_factor,
                int(population.optima_noptimal_mask[i]),
                population.optima_fitnesses[i],
            ]
            for i in range(population.num_niches)
        ]
        self.niche_metrics_printer(metrics_data)

        # Log diversity metrics
        diversity_row = self._calculate_diversity_metrics(iteration, population)
        self.diversity_printer([diversity_row])

        # Log detailed evolution of best points
        if self.detailed:
            evolution_data = [
                [
                    iteration,
                    f"nopt{i}" if i > 0 else "optimum",
                    population.optima_raw_objectives[i],
                    population.optima_violations[i] * population.violation_factor,
                    int(population.optima_noptimal_mask[i]),
                    population.optima_fitnesses[i],
                ] + population.optima_points[i].tolist()
                for i in range(population.num_niches)
            ]
            self.evolution_printer(evolution_data)

    def finalize(self, population):
        """
        Writes the final summary of n-optima and flushes all log files.

        Args:
            best_solutions (dict): The final dictionary of best solutions.
        """
        print("Finalizing logs...", end="")
        # Write final n-optima data
        niche_names = [f"nopt{i}" if i > 0 else "optimum" for i in range(len(population.optima_points))]
        final_data = np.vstack(
            [
                niche_names,
                population.optima_raw_objectives,
                population.optima_violations * population.violation_factor,
                population.optima_noptimal_mask.astype(int),
                population.optima_fitnesses,
                population.optima_points.T,
            ]
        ).T.tolist()
        self.noptima_printer(final_data)

        # Flush all remaining data in buffers
        self.niche_metrics_printer._flush()
        self.diversity_printer._flush()
        if self.detailed:
            self.evolution_printer._flush()
        self.noptima_printer._flush()
        print("Logging complete.")

    def _calculate_diversity_metrics(self, iteration: int, population) -> list:
        return [
            iteration,
            population.vesa,
            population.shannon,
            np.mean(population.stds) if population.stds.size > 0 else 0.0,
            np.min(population.stds) if population.stds.size > 0 else 0.0,
            np.max(population.stds) if population.stds.size > 0 else 0.0,
            np.mean(population.variances) if population.variances.size > 0 else 0.0,
            np.min(population.variances) if population.variances.size > 0 else 0.0,
            np.max(population.variances) if population.variances.size > 0 else 0.0,
            diversity.sum_of_fitness(population.fitnesses, population.noptimal_mask),
            population.mean_fitness,
        ]
