import warnings

import torch
from transformers import HfArgumentParser

from configs.cfg import Configs
from train_test.caption_execute import CaptionTrainerTester
from utils.services import Services
from utils.tokenizers import CustomTokenizer

warnings.filterwarnings("ignore")


def main() -> None:
    """Executes main flow."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Please ensure that you have a compatible GPU and CUDA installed."
        )

    cfg = HfArgumentParser(Configs).parse_args_into_dataclasses()[0]
    print(f"Configurations:\n{cfg}")
    Services.initialize_experiment_directories(cfg)
    Services.seed_everything(cfg.seed)
    tokenizer = CustomTokenizer(cfg)
    trainer_tester = CaptionTrainerTester(cfg, tokenizer)
    trainer_tester.execute()


if __name__ == "__main__":
    main()
