import io
import os
import re

import pandas as pd
from PIL import Image
from tqdm import tqdm

tqdm.pandas()


class PreprocessRocoDataset:
    def __init__(self) -> None:
        """Initializes PreprocessRocoDataset."""

    def _clean_report(self, report: str) -> str:
        """Cleans the report.

        Args:
            report: Report.

        Returns:
            Output: Cleaned report.
        """
        caption = report

        caption = re.sub(r"\n|\t", " ", caption)
        caption = re.sub(
            r"\b\d{1,3}[-\s]?(year[-\s]?old)\b", "xxxx", caption, flags=re.IGNORECASE
        )
        caption = re.sub(
            r"\b\d+(?:\.\d+)?\s?(mm|cm|m|kg|g|mg|ml)\b",
            "xxxx",
            caption,
            flags=re.IGNORECASE,
        )
        caption = re.sub(r"\s+", " ", caption)
        caption = caption.strip()

        if not caption.endswith("."):
            caption += "."

        caption = caption[:-1].replace(".", " . ").replace(",", " , ")
        caption = caption + " ."
        caption = caption.strip().lower()

        return caption

    def _form_df(self, split: str) -> pd.DataFrame:
        """Creates a dataframe.

        Args:
            split: Which split to use i.e. train, validation, or test.

        Returns:
            Output: Dataframe.
        """
        all_files = []
        for _, _, filename in os.walk("data/roco/parquet"):
            for f in filename:
                if split in f:
                    all_files.append(f)

        df = pd.DataFrame(columns=["image_id", "findings"])
        for filename in tqdm(all_files):
            temp_df = pd.read_parquet(f"data/roco/parquet/{filename}")
            temp_df.drop(["image"], axis=1, inplace=True)
            temp_df.rename(columns={"caption": "findings"}, inplace=True)
            temp_df["image_id"] = temp_df["image_id"].apply(
                lambda x: f"data/roco/{split}/images/{x}.png"
            )
            df = pd.concat([df, temp_df], axis=0)

        return df

    def _save_images(self, split: str) -> None:
        """Saves image bytes data onto disk.

        Args:
            split: Which split to use i.e. train, validation, or test.
        """
        all_files = []
        for _, _, filename in os.walk("data/roco/parquet"):
            for f in filename:
                if split in f:
                    all_files.append(f)

        for filename in tqdm(all_files):
            df = pd.read_parquet(f"data/roco/parquet/{filename}")
            for image_bytes_data, image_id in tqdm(
                zip(df["image"].values.tolist(), df["image_id"].values.tolist())
            ):
                image_bytes = image_bytes_data["bytes"]
                image = Image.open(io.BytesIO(image_bytes))
                image.save(f"data/roco/{split}/images/{image_id}.png")

    def preprocess(self) -> None:
        """Preprocesses the dataset."""
        train_df = self._save_images("train")
        val_df = self._save_images("validation")
        test_df = self._save_images("test")

        train_df = self._form_df("train")
        val_df = self._form_df("validation")
        test_df = self._form_df("test")

        train_df["findings"] = train_df["findings"].progress_apply(self._clean_report)
        val_df["findings"] = val_df["findings"].progress_apply(self._clean_report)
        test_df["findings"] = test_df["findings"].progress_apply(self._clean_report)

        train_df.to_csv("data/roco/train/train.csv", index=False)
        val_df.to_csv("data/roco/validation/val.csv", index=False)
        test_df.to_csv("data/roco/test/test.csv", index=False)

        train_df = pd.read_csv("data/roco/train/train.csv", encoding="utf-8").dropna()
        val_df = pd.read_csv("data/roco/validation/val.csv", encoding="utf-8").dropna()
        test_df = pd.read_csv("data/roco/test/test.csv", encoding="utf-8").dropna()

        train_df.to_csv("data/roco/train/train.csv", index=False)
        val_df.to_csv("data/roco/validation/val.csv", index=False)
        test_df.to_csv("data/roco/test/test.csv", index=False)

        print(
            f"Splits info: train_size={len(train_df)} | val_size={len(val_df)} | test_size={len(test_df)}"
        )


def main() -> None:
    """Executes the main flow."""
    PreprocessRocoDataset().preprocess()


if __name__ == "__main__":
    main()
