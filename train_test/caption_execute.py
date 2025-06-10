import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from architecture.models import CaptionModel
from configs.cfg import Configs
from utils.dataloaders import CaptionModelDataLoaders
from utils.losses import Losses
from utils.metrics import Metrics
from utils.services import Services
from utils.tokenizers import CustomTokenizer


class CaptionTrainerTester:
    def __init__(
        self,
        cfg: Configs,
        tokenizer: CustomTokenizer,
    ) -> None:
        """Initializes TrainerTester.

        Args:
            cfg: Configurations.
            tokenizer: Tokenizer.
        """
        self.device = "cuda"
        self.cfg = cfg
        self.epoch_run = 1
        self.plateau_count = 0
        self.best_val_score = -float("inf")
        self.tokenizer = tokenizer
        self.metrics = Metrics()

        self.train_loader = CaptionModelDataLoaders.get_train_dataloader(cfg, tokenizer)
        self.val_loader = CaptionModelDataLoaders.get_val_dataloader(cfg, tokenizer)
        self.test_loader = CaptionModelDataLoaders.get_test_dataloader(cfg, tokenizer)

        self.model = CaptionModel(cfg, tokenizer.get_vocab_size())
        self.xe_loss_fn = nn.CrossEntropyLoss(
            ignore_index=tokenizer.get_id_by_token("<pad>")
        )
        self.optimizer = AdamW(
            [
                {"params": self.model.vis_enc.parameters(), "lr": cfg.v_lr},
                {
                    "params": list(self.model.proj.parameters())
                    + list(self.model.text_embs.parameters())
                    + list(self.model.encoder.parameters())
                    + list(self.model.decoder.parameters())
                    + list(self.model.gen_logits.parameters()),
                    "lr": cfg.lr,
                },
            ],
            weight_decay=cfg.weight_decay,
            amsgrad=True,
        )
        self.scheduler = StepLR(
            self.optimizer, step_size=cfg.epochs // 2, gamma=cfg.gamma
        )

        print(f"Total params: {sum(p.numel() for p in self.model.parameters())}")
        print(
            f"Trainable params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
        )

    def _train(self) -> float:
        """Trains the model.

        Returns:
            Loss: Train loss.
        """
        self.model.train()
        running_train_loss = 0.0
        total_samples = len(self.train_loader)

        for _, batch in tqdm(
            enumerate(self.train_loader),
            total=total_samples,
            leave=True,
            desc="Train",
        ):
            image = batch["image"].to(self.device)  # [b, c, h, w]
            input_ids = batch["input_ids"].to(self.device)  # [b, text_seq_len]
            pad_mask = batch["pad_mask"].to(self.device)  # [b, text_seq_len]
            label_ids = batch["label_ids"].to(self.device)  # [b, text_seq_len]

            gen_logits, total_lb_loss = self.model(
                image,
                input_ids,
                pad_mask,
            )  # [b, text_seq_len, vocab_size]

            loss = Losses.compute_loss(
                self.xe_loss_fn,
                self.tokenizer.get_vocab_size(),
                gen_logits,
                label_ids,
                total_lb_loss,
            )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            running_train_loss += loss.item()

        running_train_loss /= total_samples
        self.scheduler.step()

        return running_train_loss

    @torch.no_grad()
    def _validate(self) -> tuple[float, float]:
        """Validates the model.

        Returns:
            Loss: Validation loss.
            Score: Validation metrics score.
        """
        self.model.eval()

        running_val_loss = 0.0
        total_samples = len(self.val_loader)
        sos_id = self.tokenizer.get_id_by_token("<sos>")
        eos_id = self.tokenizer.get_id_by_token("<eos>")
        pad_id = self.tokenizer.get_id_by_token("<pad>")
        generated_reports = []
        actual_reports = []

        for _, batch in tqdm(
            enumerate(self.val_loader),
            total=total_samples,
            leave=True,
            desc="Validation",
        ):
            image = batch["image"].to(self.device)  # [b, c, h, w]
            input_ids = batch["input_ids"].to(self.device)  # [b, text_seq_len]
            pad_mask = batch["pad_mask"].to(self.device)  # [b, text_seq_len]
            label_ids = batch["label_ids"].to(self.device)  # [b, text_seq_len]

            gen_logits, total_lb_loss = self.model(
                image,
                input_ids,
                pad_mask,
            )  # [b, text_seq_len, vocab_size]

            loss = Losses.compute_loss(
                self.xe_loss_fn,
                self.tokenizer.get_vocab_size(),
                gen_logits,
                label_ids,
                total_lb_loss,
            )
            running_val_loss += loss.item()

            gen_report_ids = self.model.beam_search(
                image,
                sos_id,
                eos_id,
                pad_id,
                self.cfg.beam_width,
                self.cfg.text_seq_len,
            )  # [b, text_seq_len]

            for seq in gen_report_ids:
                generated_reports.append(
                    self.tokenizer.decode_by_ids(seq.detach().cpu().numpy().tolist())
                )
            for seq in label_ids:
                actual_reports.append(
                    self.tokenizer.decode_by_ids(seq.detach().cpu().numpy().tolist())
                )

        running_val_loss /= total_samples

        preds = [[rep] for rep in generated_reports]
        refs = [[rep] for rep in actual_reports]
        mmscore = self.metrics.get_monitored_metrics_scores(preds, refs)

        if mmscore >= self.best_val_score:
            self.best_val_score = mmscore
            self.plateau_count = 0
            snapshot = {
                "model": self.model.state_dict(),
                "whole_model": self.model,
            }
            Services.save_snapshot(
                self.cfg.model_snapshot_filepath,
                self.cfg.serialized_model_snapshot_filepath,
                snapshot,
            )
        else:
            self.plateau_count += 1

        return running_val_loss, mmscore

    @torch.no_grad()
    def _test(self) -> None:
        """Tests the model."""
        snapshot = Services.load_snapshot(self.cfg.model_snapshot_filepath, self.device)
        self.model.load_state_dict(snapshot["model"])
        self.model.eval()

        total_samples = len(self.test_loader)
        sos_id = self.tokenizer.get_id_by_token("<sos>")
        eos_id = self.tokenizer.get_id_by_token("<eos>")
        pad_id = self.tokenizer.get_id_by_token("<pad>")
        generated_reports = []
        actual_reports = []

        for _, batch in tqdm(
            enumerate(self.test_loader),
            total=total_samples,
            leave=True,
            desc="Test",
        ):
            image = batch["image"].to(self.device)  # [b, c, h, w]
            label_ids = batch["label_ids"].to(self.device)  # [b, text_seq_len]

            gen_report_ids = self.model.beam_search(
                image,
                sos_id,
                eos_id,
                pad_id,
                self.cfg.beam_width,
                self.cfg.text_seq_len,
            )  # [b, text_seq_len]

            for seq in gen_report_ids:
                generated_reports.append(
                    self.tokenizer.decode_by_ids(seq.detach().cpu().numpy().tolist())
                )
            for seq in label_ids:
                actual_reports.append(
                    self.tokenizer.decode_by_ids(seq.detach().cpu().numpy().tolist())
                )

        df = pd.DataFrame(
            {
                "actual_report": actual_reports,
                "generated_report": generated_reports,
            }
        )
        df.to_csv(self.cfg.gen_reports_filepath, index=False)
        print("Saved generated reports")

        preds = [[rep] for rep in generated_reports]
        refs = [[rep] for rep in actual_reports]
        self.metrics.calculate_save_nlg_metrics_scores(
            self.cfg.nlg_scores_filepath, preds, refs
        )

    def execute(self) -> None:
        """Executes training, validation, and testing."""
        print("Training and validation starts")
        self.model.to(self.device)

        for epoch in range(self.epoch_run, self.cfg.epochs + 1):
            train_loss = self._train()
            val_loss, mmscore = self._validate()

            print(
                f"Epoch: {epoch} | "
                f"T-Loss: {train_loss:.6f} | "
                f"V-Loss: {val_loss:.6f} | "
                f"V-Score: {mmscore:.6f} | "
                f"V-Best Score: {self.best_val_score:.6f} | "
                f"Last lr: {self.scheduler.get_last_lr()[0]:.2e},{self.scheduler.get_last_lr()[1]:.2e} | "
                f"Plateau count: {self.plateau_count}"
            )

            if self.plateau_count >= self.cfg.max_patience:
                print("Early stopping triggered")
                break

        print("Training and validation complete")
        print("Testing starts")
        self._test()
        print("Testing complete")
