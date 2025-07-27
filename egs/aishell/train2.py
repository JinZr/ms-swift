import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

from swift.llm import DatasetName, ModelType, SftArguments, sft_main

# from modelscope.msdatasets import MsDataset


sft_main(
    SftArguments(
        model_type=ModelType.qwen2_audio_7b_instruct,
        model_id_or_path=None,
        # dataset=[DatasetName.aishell1_zh],
        dataset=[
            # DatasetName.aishell1_zh,
            "data/aishell_train.jsonl"
            # MsDataset.load(
            #     "speech_asr/speech_asr_aishell1_subset",
            #     subset_name="default",
            #     split="train",
            # )
            # "speech_asr/speech_asr_aishell1_trainsets:train",
        ],
        val_dataset=[
            "data/aishell_dev.jsonl#500"
            # MsDataset.load(
            #     "speech_asr/speech_asr_aishell1_subset",
            #     subset_name="default",
            #     split="validation",
            # )
            # "speech_asr/speech_asr_aishell1_trainsets:validation"
        ],
        # val_dataset=[DatasetName.aishell1_zh],
        per_device_train_batch_size=96,
        per_device_eval_batch_size=96,
        num_train_epochs=10,
        save_steps=500,
        eval_strategy="epoch",
        report_to=[
            "tensorboard",
        ],
        dataloader_num_workers=32,
        dataloader_drop_last=True,
        lazy_tokenize=True,
        dtype="fp32",
    )
)
