if [ $# -lt 1 ]; then
    echo $0 "<model_fn>"
    exit
fi

CUDA_VISIBLE_DEVICES=1 python eval_lstm.py \
    --model_fn $1 \
    --data_dir_test ./data/NTURGBD/skeletons_3d_sub_center \
    --input_size 75 \
    --seq_len 10 \
    --hidden_size 256 \
    --num_class 60 \
    --use_bias \
    --batch_size 56 \
    --num_layers 2 \
    --num_workers 4
