import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

tqdm.pandas()


class PreprocessCovctrDataset:
    def __init__(self) -> None:
        """Initializes PreprocessIuxrayDataset."""

    def _clean_report(self, report: str) -> str:
        """Cleans the report.

        Args:
            text: Report.

        Returns:
            Output: Cleaned report.
        """
        text = (
            report.replace('"', "")
            .replace("(", "")
            .replace(")", "")
            .replace(",", " ,")
            .replace("-", "")
            .replace(";", " ,")
            .replace(".", " .")
            .strip()
            .lower()
        )

        return text + " ." if text[-1] != "." else text

    def preprocess(self) -> None:
        """Preprocesses the dataset."""
        df = pd.read_csv("data/covctr/reports.csv")
        df["findings"] = df["reports_En"]
        df.drop(
            ["reports_En", "COVID", "terminologies", "impression"], axis=1, inplace=True
        )
        df["image_id"] = df["image_id"].progress_apply(
            lambda x: f"data/covctr/images/{x}"
        )
        df["findings"] = df["findings"].progress_apply(self._clean_report)

        # Data splits
        train_df, val_test_df = train_test_split(
            df, test_size=0.2, shuffle=True, random_state=9876
        )
        val_df, test_df = train_test_split(
            val_test_df, test_size=0.5, shuffle=True, random_state=9876
        )

        # Save splits
        train_df.to_csv("data/covctr/train.csv", index=False)
        val_df.to_csv("data/covctr/val.csv", index=False)
        test_df.to_csv("data/covctr/test.csv", index=False)

        train_df = pd.read_csv("data/covctr/train.csv", encoding="utf-8").dropna()
        val_df = pd.read_csv("data/covctr/val.csv", encoding="utf-8").dropna()
        test_df = pd.read_csv("data/covctr/test.csv", encoding="utf-8").dropna()

        train_df.to_csv("data/covctr/train.csv", index=False)
        val_df.to_csv("data/covctr/val.csv", index=False)
        test_df.to_csv("data/covctr/test.csv", index=False)

        print(
            f"Splits info: train_size={len(train_df)} | val_size={len(val_df)} | test_size={len(test_df)}"
        )


def main() -> None:
    """Executes the main flow."""
    PreprocessCovctrDataset().preprocess()


if __name__ == "__main__":
    main()
