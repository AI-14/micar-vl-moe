import torch
import torch.nn as nn

from configs.cfg import Configs

from .blocks import (
    Decoder,
    Encoder,
    MultiScaleVisionEncoder,
    RMSNorm,
    TeLU,
    TextEmbedding,
)
from .utils import create_causal_mask


class CaptionModel(nn.Module):
    def __init__(self, cfg: Configs, vocab_size: int) -> None:
        """Initializes CaptionModel.

        Args:
            cfg: Configurations.
            vocab_size: Vocabulary size.
        """
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size

        act_fn_options = {
            "silu": nn.SiLU,
            "gelu": nn.GELU,
            "selu": nn.SELU,
            "telu": TeLU,
        }

        self.vis_enc = MultiScaleVisionEncoder(cfg.d_v)
        self.proj = nn.Sequential(
            RMSNorm(cfg.d_v),
            nn.Linear(cfg.d_v, cfg.d_model),
        )
        self.text_embs = TextEmbedding(
            vocab_size, cfg.d_model, cfg.text_seq_len, cfg.dropout
        )
        self.encoder = Encoder(
            cfg.d_model,
            cfg.d_latent,
            cfg.d_ff,
            cfg.num_heads,
            cfg.num_layers,
            cfg.qk_nope_dim,
            cfg.qk_rope_dim,
            act_fn_options[cfg.act_fn],
            cfg.attention_dropout,
            cfg.dropout,
        )
        self.decoder = Decoder(
            cfg.d_model,
            cfg.d_latent,
            cfg.d_ff,
            cfg.num_heads,
            cfg.num_layers,
            cfg.qk_nope_dim,
            cfg.qk_rope_dim,
            act_fn_options[cfg.act_fn],
            cfg.attention_dropout,
            cfg.dropout,
            cfg.num_experts,
            cfg.k,
        )
        self.gen_logits = nn.Sequential(
            RMSNorm(cfg.d_model), nn.Linear(cfg.d_model, vocab_size, bias=False)
        )

        self._init_weights_with_xavier_uniform()

    def _init_weights_with_xavier_uniform(self) -> None:
        """Initializes weights of layers with xavier uniform."""
        for p in self.proj.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for p in self.text_embs.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for p in self.gen_logits.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        image: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_pad_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, float]:
        """Executes flow of CaptionModel.

        Args:
            image: Shape of [b, c, h, w].
            text_input_ids: Shape of [b, text_seq_len].
            text_pad_mask: Shape of [b, text_seq_len].

        Returns:
            Output: Shape of [b, text_seq_len, vocab_size].
            Loss: Total load balancing loss.
        """
        batch_size = image.shape[0]
        device = image.device

        vis_enc_out = self.vis_enc(image)  # [b, image_seq_len, d_v]
        vis_enc_out = self.proj(vis_enc_out)  # [b, image_seq_len, d_model]
        vis_enc_out = self.encoder(vis_enc_out)  # [b, image_seq_len, d_model]

        text_emb_out = self.text_embs(text_input_ids)  # [b, text_seq_len, d_model]

        mask = create_causal_mask(
            batch_size,
            text_emb_out.shape[1],
            text_pad_mask,
            device,
        )  # [b, text_seq_len, text_seq_len]

        dec_out, total_lb_loss_dec = self.decoder(
            text_emb_out, vis_enc_out, mask
        )  # [b, text_seq_len, d_model]

        logits = self.gen_logits(dec_out)  # [b, text_seq_len, vocab_size]

        return logits, total_lb_loss_dec

    def beam_search(
        self,
        image: torch.Tensor,
        sos_id: int,
        eos_id: int,
        pad_id: int,
        beam_size: int,
        max_seq_len: int,
    ) -> torch.Tensor:
        """Performs beam search decoding.

        Args:
            image: Shape of [b, c, h, w].
            sos_id: <sos> token ID.
            eos_id: <eos> token ID.
            pad_id: <pad> token ID.
            beam_size: Beam size.
            max_seq_len: Maximum sequence length.

        Returns:
            Decoded sequences: Shape of [b, max_seq_len].
        """
        batch_size = image.shape[0]  # [b]
        device = image.device

        # Encode the image features
        vis_enc_out = self.vis_enc(image)  # [b, image_seq_len, d_v]
        vis_enc_out = self.proj(vis_enc_out)  # [b, image_seq_len, d_model]
        vis_enc_out = self.encoder(vis_enc_out)  # [b, image_seq_len, d_model]

        # Initialize sequences with start token
        sequences = torch.full(
            (batch_size, beam_size, 1), sos_id, dtype=torch.long, device=device
        )  # [b, beam_size, 1]
        sequence_scores = torch.zeros(
            batch_size, beam_size, device=device
        )  # [b, beam_size]

        for step in range(max_seq_len):
            # Flatten sequences for processing
            flat_sequences = sequences.reshape(
                batch_size * beam_size, -1
            )  # [b * beam_size, step+1]

            # Generate text embeddings
            text_emb_out = self.text_embs(
                flat_sequences
            )  # [b * beam_size, step + 1, d_model]

            # Add visual features
            vis_enc_expanded = (
                vis_enc_out.unsqueeze(1)
                .expand(-1, beam_size, -1, -1)
                .reshape(-1, vis_enc_out.shape[1], vis_enc_out.shape[2])
            )  # [b * beam_size, image_seq_len, d_model]

            mask = create_causal_mask(
                batch_size * beam_size, text_emb_out.shape[1], None, device
            )  # [b * beam_size, step + 1, step + 1]

            dec_out, _ = self.decoder(
                text_emb_out, vis_enc_expanded, mask
            )  # [b * beam_size, step + 1, d_model]

            logits = self.gen_logits(dec_out[:, -1, :])  # [b * beam_size, vocab_size]
            log_probs = nn.functional.log_softmax(
                logits, dim=-1
            )  # [b * beam_size, vocab_size]

            # Update sequences and scores
            log_probs = log_probs.reshape(
                batch_size, beam_size, -1
            )  # [b, beam_size, vocab_size]
            scores = (
                sequence_scores.unsqueeze(-1) + log_probs
            )  # [b, beam_size, vocab_size]
            scores = scores.reshape(batch_size, -1)  # [b, beam_size * vocab_size]

            top_scores, top_indices = scores.topk(
                beam_size, dim=-1
            )  # [b, beam_size], [b, beam_size]
            sequence_scores = top_scores  # [b, beam_size]

            next_tokens = top_indices % self.vocab_size  # [b, beam_size]
            beam_indices = top_indices // self.vocab_size  # [b, beam_size]

            # Update sequences with the next tokens
            sequences = sequences.gather(
                1, beam_indices.unsqueeze(-1).expand(-1, -1, sequences.shape[-1])
            )  # [b, beam_size, step+1]
            sequences = torch.cat(
                [sequences, next_tokens.unsqueeze(-1)], dim=-1
            )  # [b, beam_size, step+2]

            # Check for EOS and stop early if all beams end
            eos_mask = next_tokens == eos_id  # [b, beam_size]
            if eos_mask.all():
                break

        # Pad sequences to max_seq_len with EOS token
        if sequences.shape[-1] < max_seq_len:
            pad_size = max_seq_len - sequences.shape[-1]
            pad = torch.full(
                (batch_size, beam_size, pad_size),
                pad_id,
                dtype=torch.long,
                device=device,
            )  # [b, beam_size, pad_size]
            sequences = torch.cat(
                [sequences, pad], dim=-1
            )  # [b, beam_size, max_seq_len]

        # Select the best beam (highest score) for each batch
        best_beam_indices = sequence_scores.argmax(dim=-1)  # [b]
        best_sequences = sequences.reshape(
            batch_size, beam_size, -1
        )[  # [b, beam_size, max_seq_len]
            torch.arange(batch_size, device=device), best_beam_indices
        ]  # [b, max_seq_len]

        return best_sequences
