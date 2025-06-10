import torch
import torch.nn as nn


class Losses:
    @staticmethod
    def compute_loss(
        xe_loss_fn: nn.CrossEntropyLoss,
        vocab_size: int,
        gen_logits: torch.Tensor,
        label_ids: torch.Tensor,
        total_lb_loss: float,
    ) -> float:
        """Computes total loss.
        Args:
            xe_loss_fn: XE loss function.
            vocab_size: Vocabulary size.
            gen_logits: Generated report logits. Shape of [B, text_seq_len, vocab_size].
            label_ids: Label ids. Shape of [B, text_seq_len].
            total_lb_loss: Total load balancing loss.

        Returns:
            Output: Total loss.
        """
        total_loss = (
            xe_loss_fn(
                gen_logits.contiguous().view(-1, vocab_size),
                label_ids.contiguous().view(-1),
            )
            + 0.01 * total_lb_loss
        )

        return total_loss
