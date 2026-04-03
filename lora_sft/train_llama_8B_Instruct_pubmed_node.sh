# --nnodes 1 --nproc_per_node 4 --master_port 25641


deepspeed --include localhost:0 train_instruct_sft.py \
    --deepspeed ds_zero2_no_offload.json \
    --model_name_or_path "meta-llama/Llama-3.1-8B-Instruct" \
    --use_lora true \
    --use_deepspeed true \
    --data_path "output/pubmed/node_classification/train/2_hop_without_label.jsonl" \
    --bf16 true \
    --fp16 false \
    --output_dir "output_model/llama_8B_Instruct_pubmed_node_2_hop_without_label" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 12 \
    --eval_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --learning_rate 2e-4 \
    --logging_steps 10 \
    --tf32 true \
    --model_max_length 1024

# --save_steps 1000 \

# 3B 2 1 12
# 8B 1 1 8