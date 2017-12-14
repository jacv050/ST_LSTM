
CUDA_VISIBLE_DEVICES=1 python train_lstm.py --model_path ./models_lstm_ntu_h128 \
    --data_dir ./data/NTURGBD/nturgb+d_color_skeletons_train \
    --data_dir_test ./data/NTURGBD/nturgb+d_color_skeletons_train_val \
    --input_size 50 \
    --hidden_size 128 \
    --num_class 60 \
    --use_bias \
    --log_step 100 \
    --eval_step 200 \
    --save_step 1000 \
    --batch_size 56 \
    --num_epochs 200 \
    --num_workers 4
