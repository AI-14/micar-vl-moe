import json
import os
from collections import Counter

import pandas as pd
import torch

from configs.cfg import Configs


class CustomTokenizer:
    def __init__(self, cfg: Configs) -> None:
        """Initializes CustomTokenizer.

        Args:
            cfg: Configurations.
        """
        self.cfg = cfg
        self.df = pd.read_csv(cfg.df_train_filepath, encoding="utf-8")
        self.min_freq = cfg.min_frequency
        self.text_seq_len = cfg.text_seq_len

        if os.path.exists(cfg.token2id_filepath) and os.path.exists(
            cfg.id2token_filepath
        ):
            with open(cfg.token2id_filepath, "r") as f:
                self.token2id = {token: int(idx) for token, idx in json.load(f).items()}
            with open(cfg.id2token_filepath, "r") as f:
                self.id2token = {int(idx): token for idx, token in json.load(f).items()}
            print(f"Vocabulary loaded with total size of {len(self.token2id)}")
        else:
            self.token2id, self.id2token = self._create_vocabulary()
            with open(cfg.token2id_filepath, "w") as f:
                json.dump(self.token2id, f)
            with open(cfg.id2token_filepath, "w") as f:
                json.dump(self.id2token, f)
            print(
                f"Vocabulary created and loaded with total size of {len(self.token2id)}"
            )

    def _create_vocabulary(self) -> tuple[dict[str, int], dict[int, str]]:
        """Creates vocabulary.

        Returns:
            Token2id mapping.
            Id2token mapping.
        """
        total_tokens = []
        text = self.df["findings"].values

        for f in text:
            total_tokens.extend(f.split())

        token_freq = Counter(total_tokens)
        vocabulary = [
            token for token, freq in token_freq.items() if freq >= self.min_freq
        ]
        vocabulary.sort()

        token2id = {"<sos>": 1, "<eos>": 2, "<pad>": 0, "<unk>": 3}
        id2token = {1: "<sos>", 2: "<eos>", 0: "<pad>", 3: "<unk>"}

        for idx, token in enumerate(vocabulary, start=4):
            token2id[token] = idx
            id2token[idx] = token

        return token2id, id2token

    def get_vocab_size(self) -> int:
        """Returns total vocab size.

        Returns:
            Output: Total vocab size.
        """
        return len(self.token2id)

    def get_token_by_id(self, id: int) -> str:
        """Returns the token given an id.

        Args:
            id: Id.

        Returns:
            Output: Text.
        """
        return self.id2token[id]

    def get_id_by_token(self, token: str) -> int:
        """Returns id given a token.

        Args:
            token: Text.

        Returns:
            Output: Id.
        """
        if token not in self.token2id:
            return self.token2id["<unk>"]

        return self.token2id[token]

    def decode_by_ids(self, ids: list[int]) -> str:
        """Decodes the ids into tokens.

        Args:
            ids: All the ids.

        Returns:
            Output: Text.
        """
        text = ""

        for i in ids:
            if self.get_token_by_id(i) not in ["<sos>", "<eos>", "<pad>"]:
                text += self.get_token_by_id(i) + " "

        return text.strip()

    def __call__(self, report: str) -> list[torch.Tensor]:
        """Converts the report into corresponding ids.

        Args:
            report: Report.

        Returns:
            Output: Input ids, pad masks, and label ids, each of shape [text_seq_len].
        """
        ids = []
        tokens = report.split()[: self.text_seq_len - 2]

        for token in tokens:
            ids.append(self.get_id_by_token(token))

        input_ids = torch.cat(
            [
                torch.tensor([self.get_id_by_token("<sos>")], dtype=torch.int64),
                torch.tensor(ids, dtype=torch.int64),
                torch.tensor([self.get_id_by_token("<eos>")], dtype=torch.int64),
                torch.tensor(
                    [self.get_id_by_token("<pad>")]
                    * (self.text_seq_len - len(ids) - 2),
                    dtype=torch.int64,
                ),
            ]
        )  # [text_seq_len] i.e. [<sos>, ...., <eos>, <pad>, ..., <pad>]
        pad_mask = (input_ids != self.get_id_by_token("<pad>")).type_as(
            input_ids
        )  # [text_seq_len]
        label_ids = torch.cat(
            [
                torch.tensor(ids, dtype=torch.int64),
                torch.tensor([self.get_id_by_token("<eos>")], dtype=torch.int64),
                torch.tensor(
                    [self.get_id_by_token("<pad>")]
                    * (self.text_seq_len - len(ids) - 1),
                    dtype=torch.int64,
                ),
            ]
        )  # [text_seq_len] i.e. [...., <eos>, <pad>, ..., <pad>]

        return [input_ids, pad_mask, label_ids]
