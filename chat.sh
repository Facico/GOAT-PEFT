# set it 
BASE_DIR=xxx #e.g. /home/xxx/GOAT-PEFT
OUT_DIR=xxx #e.g. /mnt/models/

set -xe
conda activate goat
lora_dirs=()

model='meta-llama/Llama-2-7b-hf'
task=wizardlm52k
totalbz=32
rank=8
alpha=16
bz=1
gacc=$(( totalbz / bz / CUDA_NUM ))
ep=5
lr=2e-5
k=2
e=8
aux=1e-3
TOT_CUDA="0,1,2,3,4,5,6,7"
CUDAs=(${TOT_CUDA//,/ })
CUDA_NUM=${#CUDAs[@]}
run_command="CUDA_VISIBLE_DEVICES=$TOT_CUDA torchrun --standalone --nnodes=1 --nproc-per-node=$CUDA_NUM "


FT(){

cd $BASE_DIR/goat

unset WANDB_MODE
if [ -n "$DEBUG" ]; then
  export WANDB_MODE=disabled
fi

lr=1e-5
lora=ft
bz=2

prj=$model-$task-${lora}-total${totalbz}dp${CUDA_NUM}bz${bz}lr${lr}
out="$OUT_DIR/$prj"

eval $run_command \
train_nlg.py \
--lora $lora \
--task $task \
--model $model \
--bz $bz \
--gacc $gacc \
--ep $ep \
--lr $lr \
--prj $prj \
--rank $rank \
--alpha $alpha \
--output $out \
--seed 0 \
--result $BASE_DIR/goat/results/alpacaeval \
--git_hash $(git rev-parse --short HEAD)

lora_dirs+=($prj)

}

MOE(){

lora=src.goat

cd $BASE_DIR/goat
export ETA=1.0
unset WANDB_MODE
if [ -n "$DEBUG" ]; then
  export WANDB_MODE=disabled
fi

prj=${model}-$task-${lora}${aux}-${k}in${e}-total${totalbz}dp${CUDA_NUM}bz${bz}lr${lr}

out="$OUT_DIR/$prj"

eval $run_command \
train_nlg.py \
--model $model \
--lora $lora \
--aux_loss_coeff=$aux \
--experts=$e \
--k $k \
--task $task \
--bz $bz \
--gacc $gacc \
--ep $ep \
--lr $lr \
--prj $prj \
--rank $rank \
--alpha $alpha \
--output $out \
--seed 0 \
--result $BASE_DIR/goat/results/alpacaeval \
--git_hash $(git rev-parse --short HEAD)

lora_dirs+=($prj)

}


MOE