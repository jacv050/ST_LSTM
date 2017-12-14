

CUDA_VISIBLE_DEVICES=0 python train_lstm_rl_com_gaussian_aug_avg_h.py --model_path ./models_lstm_rl_com_gaussian_aug_avg_h_clip_1 \
    --input_size 75 \
    --ds NTU \
    --hidden_size 100 \
    --num_class 60 \
    --use_bias \
    --log_step 10 \
    --eval_step 200 \
    --save_step 1000 \
    --batch_size 56 \
    --num_epochs 200 \
    --num_workers 4
