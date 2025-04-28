import json
import re

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

tqdm.pandas()


class PreprocessPeirgrossDataset:
    def __init__(self) -> None:
        """Initializes PreprocessPeirgrossDataset."""

    def _clean_report(self, report: str) -> str:
        """Cleans the report.

        Args:
            report: Report.

        Returns:
            Output: Cleaned report.
        """

        def _report_cleaner(t: str) -> str:
            return (
                t.replace("..", ".")
                .replace("..", ".")
                .replace("..", ".")
                .replace("1. ", "")
                .replace(". 2. ", ". ")
                .replace(". 3. ", ". ")
                .replace(". 4. ", ". ")
                .replace(". 5. ", ". ")
                .replace(" 2. ", ". ")
                .replace(" 3. ", ". ")
                .replace(" 4. ", ". ")
                .replace(" 5. ", ". ")
                .replace(". .", ".")
                .strip()
                .lower()
                .split(". ")
            )

        def _sent_cleaner(t: str) -> str:
            return re.sub(
                "[.,?;*!%^&_+():-\\[\\]{}]",
                "",
                t.replace('"', "")
                .replace("/", "")
                .replace("\\", "")
                .replace("'", "")
                .strip()
                .lower(),
            )

        tokens = [
            _sent_cleaner(sent)
            for sent in _report_cleaner(report)
            if _sent_cleaner(sent) != []
        ]
        report = " . ".join(tokens) + " ."

        return report

    def preprocess(self) -> None:
        """Preprocesses the dataset."""
        with open("data/pgross/captions.json") as f:
            data = json.load(f)
        df = pd.DataFrame.from_dict(data.items())
        df.columns = ["image_id", "findings"]
        df.dropna(inplace=True)

        df["image_id"] = df["image_id"].progress_apply(
            lambda x: f"data/pgross/images/{x}"
        )
        df["findings"] = df["findings"].progress_apply(
            lambda text: self._clean_report(text.split(":")[-1].strip())
        )

        # Data splits
        train_df, val_test_df = train_test_split(
            df, test_size=0.3, shuffle=True, random_state=1234
        )
        val_df, test_df = train_test_split(
            val_test_df, test_size=0.66, shuffle=True, random_state=1234
        )

        # Save splits
        train_df.to_csv("data/pgross/train.csv", index=False)
        val_df.to_csv("data/pgross/val.csv", index=False)
        test_df.to_csv("data/pgross/test.csv", index=False)

        train_df = pd.read_csv("data/pgross/train.csv", encoding="utf-8").dropna()
        val_df = pd.read_csv("data/pgross/val.csv", encoding="utf-8").dropna()
        test_df = pd.read_csv("data/pgross/test.csv", encoding="utf-8").dropna()

        train_df.to_csv("data/pgross/train.csv", index=False)
        val_df.to_csv("data/pgross/val.csv", index=False)
        test_df.to_csv("data/pgross/test.csv", index=False)

        print(
            f"Splits info: train_size={len(train_df)} | val_size={len(val_df)} | test_size={len(test_df)}"
        )


def main() -> None:
    """Executes the main flow."""
    PreprocessPeirgrossDataset().preprocess()


if __name__ == "__main__":
    main()
