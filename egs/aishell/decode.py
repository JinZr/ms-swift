import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from swift.llm import (
    InferRequest,
    BaseArguments,
    PtEngine,
    RequestConfig,
    safe_snapshot_download,
    get_model_tokenizer,
    get_template,
    load_dataset,
)
from swift.tuners import Swift

model_id = "Qwen/Qwen2-Audio-7B-Instruct"
lora_checkpoint = safe_snapshot_download("output/v3-20250610-230136/checkpoint-20")
args = BaseArguments.from_pretrained(lora_checkpoint)
# print(f'args.model: {args.model}')
# print(f'args.model_type: {args.model_type}')
# print(f'args.template_type: {args.template}')
# print(f'args.default_system: {args.system}')


# Load the pre-trained base model and its tokenizer
model, tokenizer = get_model_tokenizer(model_id)

# Instruct MS-SWIFT to apply LoRA to all linear layers by overriding default target modules
model = Swift.from_pretrained(model, lora_checkpoint)

# Retrieve the conversation template for formatting messages
template = get_template(model.model_meta.template, tokenizer)

# Build the PyTorch inference engine, here with batch size 1
engine = PtEngine.from_model_template(model=model, template=template, max_batch_size=1)

request_config = RequestConfig(max_tokens=512, temperature=0)

# Load the AISHELL-1 test set using ms-swift's native loader
dataset = load_dataset("speech_asr/speech_asr_aishell1_trainsets:test#10")[0]

# Prepare inference requests for AISHELL-1: transcribe each audio sample
infer_requests = [
    InferRequest(
        messages=[{"role": "user", "content": "<audio> 请将这段语音转写为文字。"}],
        audios=[entry["audios"]],
    )
    for entry in dataset
]

# Perform inference; this will iterate over the eval set if defined
responses = engine.infer(infer_requests, request_config, use_tqdm=True)

# Print out each generated response
for resp in responses:
    print(resp.choices[0].message.content)
