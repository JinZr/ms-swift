import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

from swift.llm import DatasetName, ModelType, SftArguments, sft_main

# from modelscope.msdatasets import MsDataset


sft_main(
    SftArguments(
        model_type="Qwen/Qwen2.5-Omni-7B",
        model_id_or_path="Qwen/Qwen2.5-Omni-7B",
        # dataset=[DatasetName.aishell1_zh],
        dataset=[
            # DatasetName.aishell1_zh,
            "data3/fold1_train.jsonl"
            # MsDataset.load(
            #     "speech_asr/speech_asr_aishell1_subset",
            #     subset_name="default",
            #     split="train",
            # )
            # "speech_asr/speech_asr_aishell1_trainsets:train",
        ],
        val_dataset=[
            "data3/fold1_val.jsonl"
            # MsDataset.load(
            #     "speech_asr/speech_asr_aishell1_subset",
            #     subset_name="default",
            #     split="validation",
            # )
            # "speech_asr/speech_asr_aishell1_trainsets:validation"
        ],
        # val_dataset=[DatasetName.aishell1_zh],
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        # save_steps=100,
        save_total_limit=-1,
        eval_strategy="epoch",
        save_strategy="epoch",
        output_dir="./output3/",
        report_to=[
            "tensorboard",
        ],
        dataloader_num_workers=32,
        dataloader_drop_last=False,
        lazy_tokenize=True,
        dtype="bf16",
    )
)
