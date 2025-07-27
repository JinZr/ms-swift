import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

from swift.llm import TrainArguments, sft_main

# Configure your training arguments using string IDs and hyperparameters
train_args = TrainArguments(
    model="Qwen/Qwen2-Audio-7B-Instruct",  # ModelScope model ID for the Qwen2 Audio instruct model
    train_type="lora",  # Use LoRA fine-tuning
    target_modules=["all-linear"],
    dataset=[
        "speech_asr/speech_asr_aishell1_trainsets:train"
    ],  # AISHELL-1 mini dataset, sampled to 5000 examples
    val_dataset=[
        "speech_asr/speech_asr_aishell1_trainsets:validation",
    ],
    # eval_datasets=[
    #     "speech_asr/speech_asr_aishell1_trainsets:test",
    # ],
    # subset=["train", "validation", "test"],
    torch_dtype="bfloat16",  # Mixed-p  recision type
    num_train_epochs=5,  # Number of epochs
    per_device_train_batch_size=72,  # Batch size per device
    per_device_eval_batch_size=72,  # Batch size per device
    learning_rate=1e-4,  # Learning rate
    output_dir="output",  # Directory to save checkpoints
    report_to=["tensorboard"],
    dataloader_num_workers=32,
    # You can add any other supported TrainingArguments fields here
    lora_rank=8,
    lora_alpha=32,
    freeze_vit=True,
    save_steps=20,
    save_total_limit=10,
    warmup_ratio=0.05,
)

# Launch supervised fine-tuning
sft_main(train_args)
