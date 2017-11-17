CUDA_VISIBLE_DEVICES=1 python train_lstm_ft.py --model_path ./models_lstm_utkinect_h256_2 \
    --data_dir ./data/utkinect/joints_processed_v2_sub_dir \
    --data_dir_test ./data/utkinect/joints_processed_v2_sub_dir_test \
    --ds UTKinect \
    --input_size 75 \
    --seq_len 6 \
    --hidden_size 256 \
    --num_class 10 \
    --use_bias \
    --log_step 1000 \
    --eval_step 2000 \
    --save_step 1000 \
    --batch_size 56 \
    --num_epochs 200 \
    --num_layers 2 \
    --num_workers 4

#CUDA_VISIBLE_DEVICES=1 python train_lstm_ft.py --model_path ./models_lstm_utkinect_h256_2 \
#    --data_dir ./data/utkinect/joints_processed_v2_sub_dir \
#    --data_dir_test ./data/utkinect/joints_processed_v2_sub_dir_test \
#    --ds UTKinect \
#    --model_ft_fn  models_lstm_ntu_h256_2/model-31.pkl \
#    --input_size 75 \
#    --seq_len 6 \
#    --hidden_size 256 \
#    --num_class 10 \
#    --use_bias \
#    --log_step 1000 \
#    --eval_step 2000 \
#    --save_step 1000 \
#    --batch_size 56 \
#    --num_epochs 200 \
#    --num_layers 2 \
#    --num_workers 4
