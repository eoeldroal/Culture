# CUDA_VISIBLE_DEVICES=0,1,2,3,
# accelerate launch --config_file accelerate_ds_zero3_mpo.yaml MPO.py --main_process_port 8123

from PIL import Image
from datasets import load_dataset, Image as HFImage, Dataset
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from trl import DPOTrainer
from trl import DPOConfig
from PIL import Image, ImageOps

dataset_id = "HuggingFaceH4/rlaif-v_formatted"
train_dataset, test_dataset = load_dataset(dataset_id, split=["train[:5%]", "test[:1%]"])

def ensure_rgb(example):
    # Convert the image to RGB if it's not already
    image = example["images"][0]
    if isinstance(image, Image.Image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        example["images"] = [image]
    return example

# Apply the transformation to the dataset (change num_proc depending on the available compute)
train_dataset = train_dataset.map(ensure_rgb, num_proc=8)
test_dataset = test_dataset.map(ensure_rgb, num_proc=8)

MODEL_ID = "NCSOFT/VARCO-VISION-2.0-14B"

processor = AutoProcessor.from_pretrained(MODEL_ID)
#! use_fast를 쓸 경우 문제 발생 -> Debug.md 파일을 참고할 것. 

model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map=None,                # ← DeepSpeed/Accelerate가 배치
)

ref_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map=None,                # ← DeepSpeed/Accelerate가 배치
)

training_args = DPOConfig(
    loss_type=["sigmoid", "bco_pair", "sft"],  # Loss types to combine, as used in the MPO paper
    loss_weights=[0.8, 0.2, 1.0],  # Corresponding weights, as used in the MPO paper
    bf16=True,
    gradient_checkpointing=True,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    dataset_num_proc=4,  # tokenization will use 1 processes
    dataloader_num_workers=8,  # data loading will use 8 workers
    logging_steps=10,
    save_strategy="steps",
    save_steps=10,
    eval_steps=10,  # Steps interval for evaluation
    eval_strategy="steps",
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=processor,
)

def main() :
    trainer.train()
    
if __name__ == "__main__" :
    main()