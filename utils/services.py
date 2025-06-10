import os
import random

import numpy as np
import torch

from configs.cfg import Configs


class Services:
    @staticmethod
    def initialize_experiment_directories(cfg: Configs) -> None:
        """Initializes all the experiment directories.

        Args:
            cfg: Configurations.
        """
        os.makedirs(cfg.experiment_runs_directory, exist_ok=True)
        os.makedirs(
            os.path.join(cfg.experiment_runs_directory, cfg.experiment_name),
            exist_ok=True,
        )
        print("Experiment directories initialized")

    @staticmethod
    def save_snapshot(
        model_snapshot_filepath: str,
        serialized_model_snapshot_filepath: str,
        snapshot: dict[str, any],
    ) -> None:
        """Saves snapshot.

        Args:
            model_snapshot_filepath: Path to save model snapshot.
            serialized_model_snapshot_filepath: Path to save serialized model snapshot.
            snapshot: Snapshot.
        """
        whole_model = snapshot["whole_model"]
        del snapshot["whole_model"]

        torch.save(snapshot, model_snapshot_filepath)
        torch.save(whole_model, serialized_model_snapshot_filepath)
        print("Snapshot and whole serialized model saved")

    @staticmethod
    def load_snapshot(
        model_snapshot_filepath: str, device: torch.device
    ) -> dict[str, any]:
        """Loads snapshot.

        Args:
            model_snapshot_filepath: Path to model snapshot.
            device: Cuda.

        Returns:
            Output: Snapshot.
        """
        snapshot = torch.load(model_snapshot_filepath, map_location=device)
        print("Snapshot loaded")

        return snapshot

    @staticmethod
    def seed_everything(seed: int) -> None:
        """Seeds everything so that experiments are deterministic.

        Args:
            seed: Seed value.
        """
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
