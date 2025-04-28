import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models import ResNet101_Weights

from configs.cfg import Configs

from .tokenizers import CustomTokenizer


class CaptionModelDataset(Dataset):
    def __init__(
        self,
        cfg: Configs,
        tokenizer: CustomTokenizer,
        split: str,
    ) -> None:
        """Initializes CustomDataset.

        Args:
            cfg: Configurations.
            tokenizer: Tokenizer.
            split: The split to use. Values include only "train", "val", and "test".
        """
        assert split in [
            "train",
            "val",
            "test",
        ], "split values include only 'train', 'val', and 'test'"

        self.cfg = cfg
        self.tokenizer = tokenizer
        self.transforms = ResNet101_Weights.IMAGENET1K_V2.transforms()
        self.split = split

        if split == "train":
            self.df = pd.read_csv(cfg.df_train_filepath, encoding="utf-8")
        elif split == "val":
            self.df = pd.read_csv(self.cfg.df_val_filepath, encoding="utf-8")
        else:
            self.df = pd.read_csv(self.cfg.df_test_filepath, encoding="utf-8")

    def __len__(self) -> int:
        """Calculates length of the dataset.

        Returns:
            Output: Length.
        """
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, int | torch.Tensor]:
        """Prepares single item for dataloader.

        Args:
            idx: Index.

        Returns:
            Output: Mapping with all the related content in it.
        """
        image_path = str(self.df["image_id"].iloc[idx])
        image = Image.open(image_path).convert("RGB")
        image = self.transforms(image)

        report = str(self.df["findings"].iloc[idx])
        input_ids, pad_mask, label_ids = self.tokenizer(
            report
        )  # [text_seq_len], [text_seq_len], [text_seq_len]

        return {
            "image": image,
            "input_ids": input_ids,
            "pad_mask": pad_mask,
            "label_ids": label_ids,
        }
