import pandas as pd
import numpy as np
from jury import Jury, load_metric
from RaTEScore import RaTEScore


class Metrics:
    def __init__(self) -> None:
        """Initializes Metrics."""
        self.metrics = [
            load_metric(
                "bleu", resulting_name="bleu_1", compute_kwargs={"max_order": 1}
            ),
            load_metric(
                "bleu", resulting_name="bleu_2", compute_kwargs={"max_order": 2}
            ),
            load_metric(
                "bleu", resulting_name="bleu_3", compute_kwargs={"max_order": 3}
            ),
            load_metric(
                "bleu", resulting_name="bleu_4", compute_kwargs={"max_order": 4}
            ),
            load_metric("meteor", resulting_name="meteor"),
            load_metric("rouge", resulting_name="rouge"),
        ]
        self.scorer = Jury(metrics=self.metrics, run_concurrent=False)
        self.rs = RaTEScore(batch_size=8, use_gpu=True)

    def get_monitored_metrics_scores(
        self, preds: list[list[str]], refs: list[list[str]]
    ) -> float:
        """Calculates monitored metrics scores.

        Args:
            preds: Predictions.
            refs: References i.e. ground truths.

        Returns:
            Score: Averaged metrics score.
        """
        scores = self.scorer(predictions=preds, references=refs)

        avg_score = (
            (
                (
                    scores["bleu_1"]["score"]
                    + scores["bleu_2"]["score"]
                    + scores["bleu_3"]["score"]
                    + scores["bleu_4"]["score"]
                )
                / 8
            )
            + (scores["meteor"]["score"] / 4)
            + (scores["rouge"]["rougeL"] / 4)
        )

        return avg_score

    def calculate_save_nlg_metrics_scores(
        self, save_filepath: str, preds: list[list[str]], refs: list[list[str]]
    ) -> None:
        """Computes and saves NLG metric scores.

        Args:
            save_filepath: Filepath to save scores.
            preds: Predictions.
            refs: References i.e. ground truths.
        """
        final_scores = {}
        scores = self.scorer(predictions=preds, references=refs)
        rs = self.rs.compute_score([i[0] for i in preds], [i[0] for i in refs])
        rs = np.array(rs)

        final_scores["BLEU-1"] = scores["bleu_1"]["score"]
        final_scores["BLEU-2"] = scores["bleu_2"]["score"]
        final_scores["BLEU-3"] = scores["bleu_3"]["score"]
        final_scores["BLEU-4"] = scores["bleu_4"]["score"]
        final_scores["METEOR"] = scores["meteor"]["score"]
        final_scores["ROUGE-L"] = scores["rouge"]["rougeL"]
        final_scores["RS-MEAN"] = np.mean(rs)
        final_scores["RS-STD"] = np.std(rs)

        for k, v in final_scores.items():
            print(f"{k}: {v}")

        df = pd.DataFrame.from_dict(final_scores.items())
        df.columns = ["Metric", "Score"]
        df.to_csv(save_filepath, index=False)
        print("Saved nlg scores")
