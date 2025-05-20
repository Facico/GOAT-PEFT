# set it 
BASE_DIR=xxx
OUT_DIR=xxx
cd $BASE_DIR/goat
conda activate goat
set -xe

function Train(){

if [ ! -d $BASE_DIR/dataset/~data/sun397/SUN397 ]; then
cd $BASE_DIR/dataset/~data
bash download_dataset.sh
cd -
else
echo "dataset1 downloaded"
fi
cd $BASE_DIR/goat
if [ ! -f cvdata.lock ]; then 
# download dataset use single process first to avoid ddp racing
python download_data.py && touch cvdata.lock
else
    echo "dataset2 downloaded"
fi
export ETA=1.0
unset WANDB_MODE
# export WANDB_MODE=offline
# export DEBUG=1
PORT=19987
CUDA_NUM=4
run_command="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc-per-node=$CUDA_NUM --master-port $PORT"
lora=src.goat
totalbz=512
model="openai/clip-vit-base-patch32"
rank=8
alpha=16
bz=64
gacc=$(( totalbz / bz / CUDA_NUM ))
lr=1e-4
lora=$1

declare -A eps=(
    ["Cars"]=35
    ["DTD"]=76
    ["EuroSAT"]=12
    ["GTSRB"]=11
    ["MNIST"]=5
    ["RESISC45"]=15
    ["SUN397"]=14
    ["SVHN"]=4
    ["CIFAR10"]=6
    ["CIFAR100"]=6
    ["STL10"]=60
    ["Food101"]=4
    ["Flowers102"]=147
    ["FER2013"]=10
    ["PCAM"]=1
    ["OxfordIIITPet"]=82
    ["RenderedSST2"]=39
    ["EMNIST"]=2
    ["FashionMNIST"]=5
    ["KMNIST"]=5
)

tasklist=(
    "Cars"
    "DTD"
    "EuroSAT"
    "GTSRB"
    "RESISC45"
    "SVHN"
    "SUN397"
)
for task in "${tasklist[@]}"; do

prj=$model-$task-${lora}-total${totalbz}dp${CUDA_NUM}bz${bz}lr${lr}
if [[ "$lora" == *"src"* ]]; then
    k=2
    e=8
    prj+="-${k}in${e}"
fi

out="$OUT_DIR/$prj"
ep=${eps[$task]}

init_command="--lora $lora \
--task $task \
--bz $bz \
--model $model \
--gacc $gacc \
--ep $ep \
--lr $lr \
--prj $prj \
--rank $rank \
--alpha $alpha \
--output $out \
--seed 0 \
--result $BASE_DIR/results/vit"

if [[ "$lora" == *"src"* ]]; then
    init_command+="\
    --k $k\
    --experts $e\
    --aux_loss_coeff=1e-3"
fi

eval $run_command \
train_vit.py \
$init_command


lora_dirs+=($prj)

done

}


Train $lora
