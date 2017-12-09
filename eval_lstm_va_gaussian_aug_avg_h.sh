
CUDA_VISIBLE_DEVICES=0 python eval_lstm_va_gaussian_aug_avg_h.py --model_fn models_lstm_va_gaussian_aug_avg_h/model-101.pkl \
    --input_size 75 \
    --ds NTU \
    --hidden_size 100 \
    --num_class 60 \
    --use_bias \
    --batch_size 256 \
    --num_workers 4
