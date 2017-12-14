

CUDA_VISIBLE_DEVICES=1 python train_lstm_gaussian_aug_avg.py --model_path ./models_lstm_gaussian_aug_avg_2d \
    --data_dir ./data/NTURGBD/nturgb+d_color_skeletons_train \
    --data_dir_test ./data/NTURGBD/nturgb+d_color_skeletons_train_val \
    --input_size 50 \
    --ds NTU \
    --hidden_size 100 \
    --num_class 60 \
    --use_bias \
    --log_step 100 \
    --eval_step 200 \
    --save_step 1000 \
    --batch_size 256 \
    --num_epochs 200 \
    --num_workers 4
