from torch.utils.data import DataLoader

from configs.cfg import Configs

from .datasets import CaptionModelDataset
from .tokenizers import CustomTokenizer


class CaptionModelDataLoaders:
    @staticmethod
    def get_train_dataloader(
        cfg: Configs,
        tokenizer: CustomTokenizer,
    ) -> DataLoader:
        """Prepares train dataloader.

        Args:
            cfg: Configurations.
            tokenizer: Tokenizer.

        Returns:
            Output: Train dataloader.
        """
        dataset = CaptionModelDataset(
            cfg,
            tokenizer,
            "train",
        )

        return DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
        )

    @staticmethod
    def get_val_dataloader(
        cfg: Configs,
        tokenizer: CustomTokenizer,
    ) -> DataLoader:
        """Prepares val dataloader.

        Args:
            cfg: Configurations.
            tokenizer: Tokenizer.

        Returns:
            Output: Val dataloader.
        """
        dataset = CaptionModelDataset(
            cfg,
            tokenizer,
            "val",
        )

        return DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
        )

    @staticmethod
    def get_test_dataloader(
        cfg: Configs,
        tokenizer: CustomTokenizer,
    ) -> DataLoader:
        """Prepares test dataloader.

        Args:
            cfg: Configurations.
            tokenizer: Tokenizer.

        Returns:
            Output: Test dataloader.
        """
        dataset = CaptionModelDataset(
            cfg,
            tokenizer,
            "test",
        )

        return DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
        )
