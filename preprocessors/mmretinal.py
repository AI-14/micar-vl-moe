import os
import re

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

tqdm.pandas()


class PreprocessMmretinalDataset:
    def __init__(self) -> None:
        """Initializes PreprocessMmretinalDataset."""

    def _clean_report(self, text: str) -> str:
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

        pattern = r"\b(?:same (?:person|patient|eye) as (?:fig(?:ure|s)\.|figures?)? \d{1,2}-\d{1,2}[a-zA-Z]?(?: and [a-zA-Z])?|fig(?:ure|s)?\.? \d{1,2}-\d{1,2}[a-zA-Z]?|[a-z] (?:for|is|shows))\b[.,]?"
        cleaned = re.sub(pattern, " ", text, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = re.sub(r"([^.\n])(\n|$)", r"\1.\2", cleaned)
        cleaned = cleaned.lower()

        tokens = [
            _sent_cleaner(sent)
            for sent in _report_cleaner(cleaned)
            if _sent_cleaner(sent) != []
        ]
        report = " . ".join(tokens) + " ."

        return report

    def _inject_image_extension(self, image_id: str, path: str) -> str:
        """Inserts correct file extension.

        Args:
            image_id: Image ID.
            path: Directory of images.

        Returns:
            Output: Image filepath with correct extension.
        """
        for _, _, filenames in os.walk(path):
            for filename in filenames:
                if image_id[-4:] in [".png", ",jpg"]:
                    continue
                elif image_id == filename[:-4]:
                    return f"{path}/{image_id + filename[-4:]}"

    def preprocess(self) -> None:
        """Preprocesses the dataset."""
        df_cfp = pd.read_csv("data/mmretinal/CFP_translated_v1.csv")
        df_ffa = pd.read_csv("data/mmretinal/FFA_translated_v1.csv")
        df_oct = pd.read_csv("data/mmretinal/OCT_translated_v1.csv")

        df_cfp = df_cfp.drop(columns=["cn_caption"], axis=1)
        df_cfp = df_cfp.rename(
            columns={"Image_ID": "image_id", "en_caption": "findings"}
        )
        df_cfp["image_id"] = df_cfp["image_id"].progress_apply(
            lambda x: self._inject_image_extension(x, "data/mmretinal/CFP")
        )

        df_ffa = df_ffa.drop(columns=["cn_caption"], axis=1)
        df_ffa = df_ffa.rename(
            columns={"Image_ID": "image_id", "en_caption": "findings"}
        )
        df_ffa["image_id"] = df_ffa["image_id"].progress_apply(
            lambda x: self._inject_image_extension(x, "data/mmretinal/FFA")
        )

        df_oct = df_oct.drop(columns=["cn_caption", "is_multipic"], axis=1)
        df_oct = df_oct.rename(
            columns={"Image_ID": "image_id", "en_caption": "findings"}
        )
        df_oct["image_id"] = df_oct["image_id"].progress_apply(
            lambda x: self._inject_image_extension(x, "data/mmretinal/OCT")
        )

        # Data splits
        train_cfp, val_test_cfp = train_test_split(
            df_cfp, test_size=0.2, shuffle=True, random_state=9876
        )
        val_cfp, test_cfp = train_test_split(
            val_test_cfp, test_size=0.5, shuffle=True, random_state=9876
        )
        train_ffa, val_test_ffa = train_test_split(
            df_ffa, test_size=0.2, shuffle=True, random_state=9876
        )
        val_ffa, test_ffa = train_test_split(
            val_test_ffa, test_size=0.5, shuffle=True, random_state=9876
        )
        train_oct, val_test_oct = train_test_split(
            df_oct, test_size=0.2, shuffle=True, random_state=9876
        )
        val_oct, test_oct = train_test_split(
            val_test_oct, test_size=0.5, shuffle=True, random_state=9876
        )

        train_df = pd.concat([train_cfp, train_ffa, train_oct], axis=0)
        val_df = pd.concat([val_cfp, val_ffa, val_oct], axis=0)
        test_df = pd.concat([test_cfp, test_ffa, test_oct], axis=0)

        train_df["findings"] = train_df["findings"].progress_apply(self._clean_report)
        val_df["findings"] = val_df["findings"].progress_apply(self._clean_report)
        test_df["findings"] = test_df["findings"].progress_apply(self._clean_report)

        # Save splits
        train_df.to_csv("data/mmretinal/train.csv", index=False)
        val_df.to_csv("data/mmretinal/val.csv", index=False)
        test_df.to_csv("data/mmretinal/test.csv", index=False)

        train_df = pd.read_csv("data/mmretinal/train.csv", encoding="utf-8").dropna()
        val_df = pd.read_csv("data/mmretinal/val.csv", encoding="utf-8").dropna()
        test_df = pd.read_csv("data/mmretinal/test.csv", encoding="utf-8").dropna()

        train_df.to_csv("data/mmretinal/train.csv", index=False)
        val_df.to_csv("data/mmretinal/val.csv", index=False)
        test_df.to_csv("data/mmretinal/test.csv", index=False)

        print(
            f"Splits info: train_size={len(train_df)} | val_size={len(val_df)} | test_size={len(test_df)}"
        )


def main() -> None:
    """Executes the main flow."""
    PreprocessMmretinalDataset().preprocess()


if __name__ == "__main__":
    main()
