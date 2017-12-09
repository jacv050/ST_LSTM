

CUDA_VISIBLE_DEVICES=0 python eval_lstm_gaussian_aug_avg.py --model_fn ./models_lstm_gaussian_aug_avg/model-151.pkl \
    --input_size 75 \
    --ds NTU \
    --hidden_size 100 \
    --num_class 60 \
    --use_bias \
    --batch_size 256 \
    --num_workers 4
