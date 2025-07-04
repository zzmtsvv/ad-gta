export WANDB_ENTITY=""
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_API_KEY=""


python generate_q.py \
  --seed=0 \
  --env_name="Dark-Room-9x9-v0" \
  --num_histories=64 \
  --num_episodes=20000 \
  --lr=1e-3 \
  --eps_coef=1.2 \
  --savedir="trajectories" \

python train_gta_ad.py \
  --project="ad-gta" \
  --group="vanilla" \
  --name="debug" \
  --hidden_dim=512 \
  --num_layers=4 \
  --num_heads=4 \
  --num_key_value_heads=2 \
  --rope_theta=15000 \
  --seq_len=800 \
  --attention_dropout=0 \
  --residual_dropout=0.2358428425577489 \
  --embedding_dropout=0.3738964974316483 \
  --normalize_qk=False \
  --pre_norm=True \
  --env_name="Dark-Room-9x9-v0" \
  --learning_rate=3e-4 \
  --warmup_ratio=0.05 \
  --weight_decay=0.0 \
  --clip_grad=1.0 \
  --subsample=1 \
  --batch_size=1024 \
  --update_steps=300000 \
  --num_workers=0 \
  --label_smoothing=0.0 \
  --eval_every=30000 \
  --eval_episodes=250 \
  --eval_train_goals=20 \
  --eval_test_goals=50 \
  --learning_histories_path="trajectories" \
  --train_seed=0 \
  --data_seed=0 \
  --eval_seed=1