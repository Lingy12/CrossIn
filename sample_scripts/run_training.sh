# lora finetune with alpaca as eval data
# bash scripts/run_training <ds_name> <stage> <exp_group> <prompt> <batch> <epoch> <lr>
lr=$6
lora_rank=64
lora_alpha=128
lora_trainable="p_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
# modules_to_save="null"
lora_dropout=0.05

epoch=$5
dataset_name=$1
stage=$2 # default to be 3
exp_group=$3
prompter_template=$4
gradient_accumulation_steps=16
port=$(shuf -i 1001-10000 -n 1)

# [[ -z "$run_name" ]] && { echo "Please assign a run_name"; exit 1; }

pretrained_model="./models/llama-2-7b-hf"
chinese_tokenizer_path="./models/llama-2-7b-hf"
per_device_train_batch_size=1
per_device_eval_batch_size=1
batch_size=$((gradient_accumulation_steps * per_device_train_batch_size))
dataset_root='data'
stage_id=$(echo $stage | tr " " "-")
model_name=$(basename -- "$pretrained_model")
run_name=$model_name.$dataset_name.$stage_id.$batch_size.$lr.$epoch.$prompter_template

dataset_dir=$dataset_root/$dataset_name
data_cache='.cache'
output_dir=~/scratch/$exp_group/$run_name


echo "Ouput dir = $output_dir"
echo "dataset name = $dataset_name"
echo "epoch = $epoch"
echo "batch_size = $batch_size"

start_time=$(date +%s)
echo "start time = $start_time"
#deepspeed_config_file=ds_zero2_no_offload.json
export WANDB_MODE=online
export WANDB_API_KEY="450f5f137524092429c1579743d3941e8d31ac5d"
export WANDB_PROJECT="multilang-finetuneing"
export WANDB_NAME=$run_name.$(date '+%Y%m%d-%H%M%S')
export WANDB_NOTES=$run_name
export WANDB_TAGS="$exp_group"
export WANDB_DIR="."
export WANDB_SERVICE_WAIT=300
export NCCL_IB_SL=1
export NCCL_IB_HCA="mlx5_0:1,mlx5_5:1"
export NCCL_DEBUG=INFO
export NCCL_ALGO=RING
# export CUDA_VISIBLE_DEVICES=0
#export MELLANOX_VISIBLE_DEVICES=0,1,6,7
#torchrun --nproc_per_node=8 --nnodes=2 --node_rank=$1 --master_addr=C300-05 --master_port=12345 run_clm_pt_with_peft.py \
#torchrun --nproc_per_node=8 --nnodes=2 --node_rank=$SLURM_PROCID --master_addr=C300-05 --master_port=12345 run_clm_pt_with_peft.py \
#accelerate launch --num_processes 16 --num_machines 2 --multi_gpu --mixed_precision bf16 --machine_rank $1 --main_process_ip C300-05 --main_process_port 12345 run_clm_pt_with_peft.py \
#pip install deepspeed
#pip install --upgrade transformers
torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=$port train_crosslingual.py \
   --model_name_or_path ${pretrained_model} \
    --tokenizer_name ${pretrained_model} \
    --dataset_dir ${dataset_dir} \
    --data_cache_dir ${data_cache} \
    --overwrite_cache False \
    --validation_split_percentage 0 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --seed 42 \
    --bf16 \
    --num_train_epochs $epoch \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --save_total_limit 5 \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 100\
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --gradient_checkpointing \
    --preprocessing_num_workers 16 \
    --block_size 4096 \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --enable_peft True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --optim "adamw_torch" \
    --torch_dtype bfloat16 \
    --deepspeed ./deepspeed/ds_mini.json \
    --flash_attn False \
    --torch_compile True \
    --report_to wandb \
    --ddp_find_unused_parameters False \
    --supported_languages en,zh,vi,es \
    --stage $stage \
    --prompter_template $prompter_template
#--gradient_checkpointing \
#--fsdp "full_shard auto_wrap" \
#--fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
