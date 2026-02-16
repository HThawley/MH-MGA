import numpy as np
from pathlib import Path
from csv import writer
from shutil import copyfile
from os import remove, mkdir
from os.path import exists
from collections.abc import Collection

from mga.commons.types import DEFAULTS
INT, FLOAT = DEFAULTS
from mga.metrics import diversity  # noqa: E402


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
        self.buffer = None

        if not resume:
            self._create_file(header, create_dir)
        elif not exists(self.file_name):
            self._create_file(header, create_dir)

    def __call__(self, data_array: np.ndarray):
        """
        Adds data to the buffer and writes it out if the save frequency is met.
        """
        self.call_count += 1
        if self.buffer is None:
            self.buffer = np.atleast_2d(data_array)
        else:
            self.buffer = np.concatenate((self.buffer, np.atleast_2d(data_array)), axis=0)
        if self.save_freq > 0:
            if self.call_count % self.save_freq == 0:
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
        if self.buffer is not None:
            print("\rWriting logs to disk... Do not interrupt.", end="")
            self._copy_and_replace()
            print("\r" + " " * 50, end="\r")
            self.buffer = None

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
        self.nobjective_printer = FilePrinter(
            file_name=f"{file_prefix}-nobjective.csv", save_freq=save_freq, resume=resume, create_dir=create_dir
        )
        self.noptimality_printer = FilePrinter(
            file_name=f"{file_prefix}-noptimality.csv", save_freq=save_freq, resume=resume
        )
        self.nfitness_printer = FilePrinter(file_name=f"{file_prefix}-nfitness.csv", save_freq=save_freq, resume=resume)
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
            if ndim > 0:
                self.evolution_printer = FilePrinter(
                    file_name=f"{file_prefix}-evolution.csv",
                    save_freq=save_freq,
                    header=["iter", "niche_id", "objective", "noptimal", "fitness"]
                    + [f"x_{i}" for i in range(ndim)],
                    resume=resume,
                )
            else:
                self.evolution_printer = FilePrinter(
                    file_name=f"{file_prefix}-evolution.csv",
                    save_freq=save_freq,
                    header=["iter", "niche_id", "objective", "noptimal", "fitness", "..."],
                    resume=resume,
                )

        self.noptima_printer = FilePrinter(
            file_name=f"{file_prefix}-noptima.csv",
            save_freq=-1,  # Only writes once at the end
            header=None,
            resume=False,  # Always overwrite final results
        )

    def log_iteration(self, iteration: int, population):
        """
        Logs all relevant data for the current iteration.

        Args:
            iteration (int): The current iteration number.
            population (Population): The population object containing current state.
        """
        self.nobjective_printer(population.current_optima_obj)
        self.noptimality_printer(population.current_optima_nop.astype(int))
        self.nfitness_printer(population.current_optima_fit)

        # Log diversity metrics
        diversity_row = self._calculate_diversity_metrics(iteration, population)
        self.diversity_printer(diversity_row)

        # Log detailed evolution of best points
        if self.detailed:
            evolution_data = np.column_stack(
                [
                    np.full(population.num_niches, iteration),
                    [f"nopt{i}" for i in range(population.num_niches)],
                    population.current_optima_obj,
                    population.current_optima_fit,
                    population.current_optima,
                ]
            )
            self.evolution_printer(evolution_data)

    def finalize(self, population):
        """
        Writes the final summary of n-optima and flushes all log files.

        Args:
            best_solutions (dict): The final dictionary of best solutions.
        """
        print("Finalizing logs...")
        # Write final n-optima data
        niche_names = ["optimum"] + [f"nopt{i}" for i in range(1, len(population.current_optima))]
        final_data = np.vstack(
            [
                niche_names,
                population.current_optima_obj,
                population.current_optima_nop.astype(int),
                population.current_optima.T,
            ]
        )
        self.noptima_printer(final_data)

        # Flush all remaining data in buffers
        self.nobjective_printer._flush()
        self.noptimality_printer._flush()
        self.nfitness_printer._flush()
        self.diversity_printer._flush()
        if self.detailed:
            self.evolution_printer._flush()
        self.noptima_printer._flush()
        print("Logging complete.")

    def _calculate_diversity_metrics(self, iteration: int, population) -> np.ndarray:
        return np.array(
            [
                [
                    iteration,
                    population.vesa,
                    population.shannon,
                    np.mean(population.stds) if population.stds.size > 0 else 0.0,
                    np.min(population.stds) if population.stds.size > 0 else 0.0,
                    np.max(population.stds) if population.stds.size > 0 else 0.0,
                    np.mean(population.variances) if population.variances.size > 0 else 0.0,
                    np.min(population.variances) if population.variances.size > 0 else 0.0,
                    np.max(population.variances) if population.variances.size > 0 else 0.0,
                    diversity.sum_of_fitness(population.fitnesses, population.is_noptimal),
                    population.mean_fitness,
                ]
            ]
        )
