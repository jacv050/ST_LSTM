

CUDA_VISIBLE_DEVICES=1 python train_lstm_va_gaussian_aug.py --model_path ./models_lstm_va_gaussian_aug \
    --input_size 75 \
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
