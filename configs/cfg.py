from dataclasses import dataclass, field


@dataclass
class Configs:
    experiment_runs_directory: str = field(default="experiment_runs")
    experiment_name: str = field(default="covctr")
    dataset_name: str = field(default="covctr")

    df_train_filepath: str = field(default="data/covctr/train.csv")
    df_val_filepath: str = field(default="data/covctr/val.csv")
    df_test_filepath: str = field(default="data/covctr/test.csv")

    token2id_filepath: str = field(default="experiment_runs/covctr/token2id.json")
    id2token_filepath: str = field(default="experiment_runs/covctr/id2token.json")
    min_frequency: int = field(default=3)

    d_v: int = field(default=2048)
    num_heads: int = field(default=8)
    num_layers: int = field(default=2)
    d_model: int = field(default=512)
    d_latent: int = field(default=768)
    qk_nope_dim: int = field(default=48)
    qk_rope_dim: int = field(default=48)
    d_ff: int = field(default=2048)
    act_fn: str = field(default="silu")
    attention_dropout: float = field(default=0.12)
    dropout: float = field(default=0.1)
    num_experts: int = field(default=8)
    k: int = field(default=2)
    text_seq_len: int = field(default=80)
    batch_size: int = field(default=8)
    epochs: int = field(default=100)
    max_patience: int = field(default=50)
    beam_width: int = field(default=3)

    model_snapshot_filepath: str = field(
        default="experiment_runs/covctr/model_snapshot.pth"
    )
    serialized_model_snapshot_filepath: str = field(
        default="experiment_runs/covctr/serialized_model.pth"
    )
    nlg_scores_filepath: str = field(default="experiment_runs/covctr/nlg_scores.csv")
    gen_reports_filepath: str = field(default="experiment_runs/covctr/gen_reports.csv")

    lr: float = field(default=1e-4)
    v_lr: float = field(default=5e-5)
    weight_decay: float = field(default=5e-5)
    gamma: float = field(default=0.1)

    seed: int = field(default=123456)
