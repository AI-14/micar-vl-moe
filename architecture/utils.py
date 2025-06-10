import torch


def create_causal_mask(
    batch_size: int,
    seq_len: int,
    pad_mask: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    """Creates causal mask.

    Args:
        batch_size: Batch size.
        seq_len: Sequence length.
        pad_mask: Shape of [b, seq_len].
        device: Cuda.

    Returns:
        Output: Shape of [b, seq_len, seq_len].
    """
    causal_mask = torch.tril(
        torch.ones(
            (seq_len, seq_len),
            dtype=torch.int64,
            device=device,
        )
    ).unsqueeze(0)  # [1, seq_len, seq_len]
    causal_mask = causal_mask.expand(batch_size, -1, -1)  # [b, seq_len, seq_len]

    if pad_mask is not None:
        pad_mask = pad_mask.unsqueeze(1)  # [b, 1, seq_len]
        causal_pad_mask = causal_mask & pad_mask  # [b, seq_len, seq_len]

        return causal_pad_mask

    return causal_mask
