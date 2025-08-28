# CUDA_VISIBLE_DEVICES=0,1,2,3
# accelerate launch --config_file accelerate_ds_zero3_sft.yaml SFT.py --main_process_port 8123
import os
import torch
import wandb
from datasets import load_dataset
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

from trl import (
    SFTConfig,
    SFTTrainer,
)

if int(os.environ.get("LOCAL_RANK", 0)) == 0:
    # wandb 초기화
    wandb.init(
        project="varco-14b-sft",  # 프로젝트 이름
        name="varco-vision-experiment-1",  # 실험 이름
        config={
            "model": "NCSOFT/VARCO-VISION-2.0-14B",
            "dataset": "HuggingFaceH4/llava-instruct-mix-vsft",
            "learning_rate": 6e-6,
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
        }
    )

MODEL_ID = "NCSOFT/VARCO-VISION-2.0-14B"
DATASET = "HuggingFaceH4/llava-instruct-mix-vsft"
OUTPUT_DIR = "runs/varco-14b-sft"

################
# Dataset
################

def collate_fn(examples):
    # Get the texts and images, and apply the chat template
    texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
    
    # ✅ 핵심 수정: 명시적으로 RGB 변환
    images = []
    for example in examples:
        img = example["images"][0]
        # PIL 이미지를 명시적으로 RGB로 변환
        rgb_img = img.convert('RGB')
        images.append(rgb_img)
    
    # Tokenize the texts and process the images
    batch = processor(images=images, text=texts, return_tensors="pt", padding=True)

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels

    return batch

dataset = load_dataset(DATASET)

args = SFTConfig(
    output_dir=OUTPUT_DIR,
    bf16=True, fp16=False,
    learning_rate=6e-6, weight_decay=0.1,

    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,

    max_steps = 48,
    logging_steps=4,
    save_strategy="steps",
    save_steps = 16,
    eval_strategy = "steps",
    eval_steps = 16,
    
    max_length=8192, # 32768은 너무 커서 줄임
    # 메모리 관련 문제가 발생함.
    #? adamw_8bit로 수정한 이후, 다시 늘릴 수도? 이때, 데이터셋이 이러한 토큰 소요량이 많은지 살펴봐야 할 듯하다.
    remove_unused_columns=False,
    dataset_kwargs={"skip_prepare_dataset": True},
    torch_empty_cache_steps=4,  # 4 스텝마다 캐시 정리 -> 학습이 진행되면서 발생하는 포화 현상을 방지할 수 있다. 

    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    ddp_find_unused_parameters=False,

    optim="adamw_8bit",
    report_to="wandb",
    run_name="varco-vision-sft-experiment",  # 실행 이름
    logging_dir="./logs",  # 로그 디렉토리
)

eval_split = "validation" if "validation" in dataset else ("test" if "test" in dataset else None)
eval_dataset = dataset[eval_split].select(range(50)) #! 이때 작은 step 수를 바탕으로 테스트만 해 보는 것이므로, eval_dataset도 이에 맞게 작은 규모로 구성해야 한다. 

################
# Model, Tokenizer & Processor
################

model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map=None,                # ← DeepSpeed/Accelerate가 배치
)

processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast = True)

################
# Training
################

trainer = SFTTrainer(
    model=model,
    args=args,
    data_collator=collate_fn,
    train_dataset=dataset["train"],
    eval_dataset=eval_dataset,
    processing_class=processor,
)

trainer.train()

trainer.save_model(OUTPUT_DIR)