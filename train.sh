#CUDA_VISIBLE_DEVICES=1 python train.py --model_path ./models_ntu_h128 \
cd /bscratch/quyou/src/ST_LSTM
echo `pwd`
python train.py --model_path ./models_ntu_h128_sample_val \
    --data_dir ./data/NTURGBD/skeletons_3d_sub_center_train \
    --data_dir_test ./data/NTURGBD/skeletons_3d_sub_center_train_val \
    --seq_len 10 \
    --ds NTU \
    --hidden_size 128 \
    --num_class 60 \
    --use_bias \
    --log_step 100 \
    --eval_step 200 \
    --save_step 1000 \
    --batch_size 56 \
    --num_epochs 200 \
    --num_workers 4
