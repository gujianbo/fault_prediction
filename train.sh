
nohup python train_lstm.py \
  --train_file /data/ett/ETTm2_feature.train \
  --valid_file /data/ett/ETTm2_feature.test \
  --model_path /data/models/ett_lstm_model \
  --log_file /data/models/lstm.log \
  --input_dim 22 \
  --hidden_dim 128 \
  --num_layers 5 \
  --dropout 0.2 \
  --max_seq_len 500 \
  --log_step 10 \
  --eval_step 20 \
  --save_step 20 > /data/models/log 2>&1 &
