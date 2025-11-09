BASE_DIR=xxx #e.g. /home/xxx/GOAT-PEFT
OUT_DIR=xxx #e.g. /mnt/models/
cd $BASE_DIR

set -xe

TOT_CUDA="0,1,2,3,4,5,6,7"
CUDAs=(${TOT_CUDA//,/ })
CUDA_NUM=${#CUDAs[@]}
run_command="CUDA_VISIBLE_DEVICES=$TOT_CUDA torchrun --standalone --nnodes=1 --nproc-per-node=$CUDA_NUM "
if ! python -c "import remote_pdb" &> /dev/null; then
    pip3 install matplotlib remote-pdb seaborn nvitop multiprocess
fi

model='meta-llama/Llama-2-7b-hf'
totalbz=16
rank=32
alpha=64
bz=2
gacc=$(( totalbz / bz / CUDA_NUM ))
ep=1
lr=1e-4
k=2
e=8
aux=1e-3
# lora=lora
module=qkvud

cd $BASE_DIR/goat
conda activate goat

MOE(){

cd $BASE_DIR/goat
#best eta=0.03 rho=40
export ETA=0.1
export RHO=20
unset WANDB_MODE
if [ -n "$DEBUG" ]; then
  export WANDB_MODE=disabled
fi

lora=src.goat
for task in commonsense170k; do

prj=$model-$task-${lora}a${aux}-${k}in${e}-total${totalbz}dp${CUDA_NUM}bz${bz}lr${lr}
out="$OUT_DIR/$prj"

eval $run_command \
train_nlg.py \
--model $model \
--lora $lora \
--aux_loss_coeff=$aux \
--experts=$e \
--k $k \
--model $model \
--task $task \
--bz $bz \
--gacc $gacc \
--ep $ep \
--lr $lr \
--prj $prj \
--rank $rank \
--alpha $alpha \
--output $out \
--modules $module \
--seed 0 \
--result $OUT_DIR/goat/results/qa \
--git_hash $(git rev-parse --short HEAD)

lora_dirs+=($prj)

done

}

EVAL(){

lora=src.goat
tasks=(
"boolq"
"piqa"
"siqa"
"hellaswag"
"winogrande"
"arc_e"
"arc_c"
"obqa"
)
#save path, e.g. meta-llama/xxxx
lora_dirs=(
xxxx
)
for lora_dir in "${lora_dirs[@]}";do
ckpt_count=$(ls "$OUT_DIR/$lora_dir" 2>/dev/null | grep -c checkpoint)
echo "Detected $ckpt_count .ckpt files in $OUT_DIR/$lora_dir"
done
for item in "${tasks[@]}"; do
    task=commonsense170k-$item
    for lora_dir in "${lora_dirs[@]}";do
    for ckpt in "$MNT/$lora_dir"/*; do
        command="$run_command eval_nlg.py \
        --task $task \
        --output $ckpt \
        --result $OUT_DIR/goat/results/qa \
        --bz 16"
        eval $command
    done
    done
done


MOE

EVAL