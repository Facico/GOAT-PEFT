# set it 
BASE_DIR=xxx #e.g. /home/xxx/GOAT-PEFT
OUT_DIR=xxx #e.g. /mnt/models/
cd $BASE_DIR

TOT_CUDA="0,1,2,3,4,5,6,7"
CUDAs=(${TOT_CUDA//,/ })
CUDA_NUM=${#CUDAs[@]}
run_command="CUDA_VISIBLE_DEVICES=$TOT_CUDA torchrun --standalone --nnodes=1 --nproc-per-node=$CUDA_NUM "

set -xe
conda activate goat
cd $BASE_DIR/goat

MOE() {
export ETA=1.0
lora=src.goat
totalbz=256
model=roberta-large
# rank=8
# alpha=16
rank=32
alpha=64
bz=${bz:-32}
gacc=$(( totalbz / bz / CUDA_NUM ))
ep=10
lr=1e-4
k=${k:-2}
e=8
aux=1e-3

unset WANDB_MODE
if [ -n "$DEBUG" ]; then
  export WANDB_MODE=disabled
fi

if [[ $lora == *"ft"* ]]; then
    lr=1e-5 
elif [[ "$task" == *"rte"* ]]; then
    lr=2e-5
else
    lr=1e-4
fi
for task in mrpc rte cola sst2 qnli mnli qqp ; do
    if [[ "$lora" == *"moe"* ]]; then
        prj=$model-$task-${lora}a${aux}-${k}in${e}-total${totalbz}dp${CUDA_NUM}bz${bz}lr${lr}
    else
        prj=$model-$task-${lora}-total${totalbz}dp${CUDA_NUM}bz${bz}lr${lr}
    fi

    if [[ "$lora" == *"lora"* ]]; then
        prj+="r${rank}a${alpha}"
    fi

    if [[ "$task" == *"rte"* ]]; then
        ep=50
    else
        ep=10
    fi
    out="$OUT_DIR/$prj"

    eval $run_command \
        train_nlu.py \
        --lora $lora \
        --task glue-mlm-$task \
        --bz $bz \
        --model $model \
        --gacc $gacc \
        --ep $ep \
        --aux_loss_coeff=$aux \
        --experts=$e \
        --k $k \
        --lr $lr \
        --prj $prj \
        --rank $rank \
        --alpha $alpha \
        --output $out \
        --seed 0 \
        --result $BASE_DIR/goat/results/glue \
        --git_hash $(git rev-parse --short HEAD) 

    lora_dirs+=($prj)
done

}

MOE
