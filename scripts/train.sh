SAVE_PATH=checkpoints/Qwen2.5-Math-7B-SCAN-PRO
MODEL=Qwen/Qwen2.5-Math-7B-Instruct
DATA_PATH=datasets/SCAN-Pro

mkdir -p $SAVE_PATH

exec > >(tee ${SAVE_PATH}/output.log) 2>&1

deepspeed --module src.train_prm.main \
    --do_train --do_eval \
    --model_path $MODEL  \
    --data_path $DATA_PATH \
    --input_key question \
    --step_key steps \
    --step_label_key scores \
    --max_length 8192 \
    --output_dir $SAVE_PATH \
    --logging_steps 1 \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 2 \
    --save_only_model true \
    --metric_for_best_model eval_f1 \
    --greater_is_better true \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --bf16 \
    --num_train_epochs 1 \
    --learning_rate 7e-6 \
    --lr_scheduler_type constant \
    --report_to tensorboard \
    --deepspeed deepspeed_configs/zero1.json
