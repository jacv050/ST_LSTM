

CUDA_VISIBLE_DEVICES=1 python train_lstm.py --model_path ./models_lstm_ntu_h256_2\
    --data_dir ./data/NTURGBD/skeletons_3d_sub_center_train \
    --ds UTKinect \
    --data_dir_test ./data/NTURGBD/skeletons_3d_sub_center_train_val \
    --input_size 75 \
    --seq_len 10 \
    --hidden_size 256 \
    --num_class 60 \
    --use_bias \
    --log_step 100 \
    --eval_step 200 \
    --save_step 1000 \
    --batch_size 56 \
    --num_epochs 200 \
    --num_layers 2 \
    --num_workers 4
