## For debugging purposes
python --version
python -m torch.utils.collect_env

## Variables 
experiment_runs_directory=experiment_runs
experiment_name=mmretinal
dataset_name=mmretinal

# Run experiment
python main.py \
--experiment_runs_directory $experiment_runs_directory \
--experiment_name $experiment_name \
--dataset_name $dataset_name \
--df_train_filepath data/$dataset_name/train.csv \
--df_val_filepath data/$dataset_name/val.csv \
--df_test_filepath data/$dataset_name/test.csv \
--token2id_filepath $experiment_runs_directory/$experiment_name/token2id.json \
--id2token_filepath $experiment_runs_directory/$experiment_name/id2token.json \
--min_frequency 3 \
--d_v 2048 \
--num_heads 8 \
--num_layers 3 \
--d_model 512 \
--d_latent 768 \
--qk_nope_dim 48 \
--qk_rope_dim 48 \
--d_ff 2048 \
--act_fn gelu \
--attention_dropout 0.12 \
--dropout 0.1 \
--num_experts 8 \
--k 2 \
--text_seq_len 60 \
--batch_size 32 \
--epochs 100 \
--max_patience 50 \
--beam_width 3 \
--model_snapshot_filepath $experiment_runs_directory/$experiment_name/model_snapshot.pth \
--serialized_model_snapshot_filepath $experiment_runs_directory/$experiment_name/serialized_model.pth \
--nlg_scores_filepath $experiment_runs_directory/$experiment_name/nlg_scores.csv \
--gen_reports_filepath $experiment_runs_directory/$experiment_name/gen_reports.csv \
--lr 1e-4 \
--v_lr 5e-5 \
--weight_decay 5e-5 \
--gamma 0.1 \
--seed 115111211