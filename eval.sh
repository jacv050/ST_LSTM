#CUDA_VISIBLE_DEVICES=1 python train.py --model_path ./models_ntu_h128 \

if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_fn>"
    exit
fi
fn=$1
python eval.py --model_fn $fn \
    --data_dir_test ./data/NTURGBD/skeletons_3d_sub_center\
    --seq_len 10 \
    --ds NTU \
    --hidden_size 128 \
    --num_class 60 \
    --use_bias \
    --batch_size 56 \
    --num_workers 8
