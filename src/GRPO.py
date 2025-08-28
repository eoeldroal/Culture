# CUDA_VISIBLE_DEVICES=0,1 trl vllm-serve --model NCSOFT/VARCO-VISION-2.0-14B --dtype bfloat16 --gpu-memory-utilization 0.85 --port 8005 --data_parallel_size 2 --enable_prefix_caching true
# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 accelerate launch --config_file accelerate_ds_zero3_grpo.yaml GRPO.py --main_process_port 8123
import os
import torch
import wandb
from datasets import load_dataset
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
import re
from trl import GRPOConfig
from trl import GRPOTrainer

from math_verify import LatexExtractionConfig, parse, verify
from latex2sympy2_extended import NormalizationConfig
from typing import Optional

MODEL_ID = "NCSOFT/VARCO-VISION-2.0-14B"
DATASET = "lmms-lab/multimodal-open-r1-8k-verified"
OUTPUT_DIR = "runs/varco-14b-grpo"

if int(os.environ.get("LOCAL_RANK", 0)) == 0:
    # wandb 초기화
    wandb.init(
        project="varco-14b-grpo",  # 프로젝트 이름
        name="varco-vision-experiment-grpo-1",  # 실험 이름
        config={
            "model": MODEL_ID,
            "dataset": DATASET,
            "learning_rate": 6e-6,
            "batch_size": 1,
            "gradient_accumulation_steps":128,
        }
    )
    
# dataset = load_dataset(DATASET, split="train[:5%]")
dataset = load_dataset(DATASET, split="train[:60%]")

split_dataset = dataset.train_test_split(test_size=0.2, seed=42)

train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]

processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True, padding_side="left")

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

# {'image': <PIL.PngImagePlugin.PngImageFile image mode=RGBA size=716x200 at 0x73B82C9A3A30>, 
# 'problem': 'Based on the image, determine the constant term after combining all the polynomial expressions representing the side lengths of the triangle. Choose the correct answer from the options provided.\n\nChoices:\nA. 3\nB. 5\nC. 8\nD. 13', 
# 'solution': "<think>Let's examine the polynomial expressions given for each side of the triangle. The side labeled \\(4x^2 + x\\) does not have a constant term. The side labeled \\(2x + 3\\) has a constant term of 3. The side labeled \\(4x^3 + 2x^2 + 5\\) has a constant term of 5. To find the total constant term, we need to add the constant terms from these expressions. So, we add 3 and 5 together. 3 + 5 = 8</think>\n\n<answer>The correct answer is C</answer>", 
# 'original_question': 'According to the question shown in the image, please first perform reasoning, then finally select the right answer from the choices, e.g., Answer: xxx.\nQuestion: Based on the image, find the constant term after combining the side lengths.\nChoices:\nA. 3\nB. 5\nC. 8\nD. 13', 
# 'original_answer': 'The constant terms from the sides $2 x + 3$ and $4 x^3 + 2 x^2 + 5$ are combined as $3 + 5 = 8$. So the answer is C\nAnswer: C'}

def make_conversation(example):
    conversation = [
        {
            "role": "system", 
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]  # 리스트와 딕셔너리 형태로 변경
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": example["problem"]},
            ],
        },
    ]
    
    # 이미지 변환
    # GRPO는 여기서처럼 이미지를 변환한다.
    img = example["image"]
    rgb_img = img.convert('RGB')
    
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True,)
    
    return {
        "prompt": prompt,
        "image": rgb_img,
    }

train_dataset = train_dataset.map(make_conversation)

train_dataset = train_dataset.remove_columns(["problem", "original_question", "original_answer"])

test_dataset = test_dataset.map(make_conversation)

test_dataset = test_dataset.remove_columns(["problem", "original_question", "original_answer"])

model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map=None,                # ← DeepSpeed/Accelerate가 배치
)

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
    rewards = [1.0 if match else 0.0 for match in matches]
    return rewards

def accuracy_reward(completions, solution, **kwargs) :
    """Reward function that checks if the completion matches the ground truth.
    - If both gold and prediction are parseable → use math verification.
    - If not parseable → compare as normalized text.
    """
    rewards = []

    for completion, sol in zip(completions, solution):
        try:
            gold_parsed = parse(sol, extraction_mode="first_match")
        except Exception as e:
            gold_parsed = []

        if len(gold_parsed) != 0:
            # Try parsing predicted answer too
            try:
                answer_parsed = parse(
                    completion,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                boxed="all",
                                units=True,
                            ),
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode="first_match",
                )
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {completion}, gold: {sol}")
                reward = None
        else:
            # fallback to text match
            reward = float(completion.strip().lower() == sol.strip().lower())

        rewards.append(reward)

    return rewards

training_args = GRPOConfig(
    use_liger_loss=True,
    #! 메모리 오류 원인 예측 3. linger_loss 써보기. 
    mask_truncated_completions=True,
    
    output_dir=OUTPUT_DIR,
    bf16=True,
    learning_rate=6e-6, 
    weight_decay=0.1,
    
    remove_unused_columns=False,  # to access the solution column in accuracy_reward
    num_train_epochs=1,
    
    # Parameters that control the data preprocessing
    per_device_train_batch_size=1,
    max_completion_length=2048,
    num_generations=4,
    max_prompt_length=8192,
    # 4096으로 하면 오류가 터진다. 
    torch_empty_cache_steps=1,
    # 최대한 캐시를 자주 정리하도록. 
    gradient_accumulation_steps=128,
    
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs = {"use_reentrant": False},
    
    # Qwen 3 모델의 KV 캐시와 충돌함.
    # 현재 메모리 병목의 원인이 Activations 때문인 것 같고, 따라서 gradient checkpointing을 활성화하는 것은 불가피한 것 같다. 
    # "Caching is incompatible with gradient checkpointing in Qwen3DecoderLayer. Setting `past_key_value=None`." 이러한 오류가 계속 뜬다. 
    # 따라서 KV 캐시를 사용할 수 밖에 없는 지금 상황에 이러한 Gradient checkpointing은 사용하기가 어려울 것 같다. 
    
    #! 메모리 오류 원인 예측 2. vllm에서 생성을 전담하므로, KV 캐시 기능을 꺼도 무방하다. 따라서 Gradient checkpointing을 써도 되지 않나?
    # -- 맞아요—GRPO에서 “생성”을 vLLM 서버가 전담하고, 학습 쪽(로그프롭·엔트로피 계산 등)만 HF 모델로 돌린다면, 훈련용 HF 모델의 use_cache(KV 캐시)는 꺼두는 게 정석입니다.
    # -- KV 캐시는 ‘증분 생성(incremental decoding)’ 가속용입니다. 한 토큰씩 생성할 때 이전 스텝의 K/V를 재사용하려고 보존하는 버퍼라서, 병렬로 정답 시퀀스를 넣는 훈련/로그프롭 계산(teacher forcing) 에서는 재사용할 “다음 스텝”이 없어 효과가 없습니다. 그래서 훈련에선 끄라고 안내합니다.
    # 즉, KV 캐시는 SFT, GRPO등 단계에서 모두 꺼도 된다! 라는 뜻이다. 
    
    use_vllm=True, 
    vllm_mode='server',
    vllm_server_port=8005,
    
    logging_steps=1,
    save_strategy="steps",
    save_steps = 16,
    eval_strategy = "steps",
    eval_steps = 16,
    
    # Parameters related to reporting and saving
    optim="adamw_torch",
    #! 메모리 오류 원인 예측 1. optimizer가 호환이 안 되나?
    # bitsandbytes와 deepspeed가 호환이 안 되어서 zero-3가 제대로 적용이 안 된 것일수도.
    # 따라서 adamw_8bit 대신 adamw_torch를 사용함
    # try 1. 
    report_to="wandb",
    run_name="varco-vision-grpo-experiment",  # 실행 이름
    logging_dir="./logs",  # 로그 디렉토리
    
)
#! 메모리 오류 원인 예측 2. vllm에서 생성을 전담하므로, KV 캐시 기능을 꺼도 무방하다. 따라서 Gradient checkpointing을 써도 되지 않나?
model.config.use_cache = False

trainer = GRPOTrainer(
    model=model,
    processing_class=processor,
    reward_funcs=[format_reward, accuracy_reward],
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()
trainer.save_model(training_args.output_dir)