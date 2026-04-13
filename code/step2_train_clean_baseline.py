# step2_train_clean_baseline.py
# 仅仅训练baseline
import os
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# ================== 配置 ==================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/autodl-tmp/huggingface_cache"

BASE_DIR = Path("/root/autodl-tmp/icml")
DATA_FILE = BASE_DIR / "data" / "clean_mixed_data.jsonl"
MODEL_NAME = "/root/autodl-tmp/models/Mistral-7B-v0.3"
OUTPUT_DIR = BASE_DIR / "outputs" / "Clean_Baseline"

print("=== 阶段 2: 训练 Clean Baseline (终极稳定兼容版) ===")

# ================== 1. 加载 Tokenizer 和 模型 ==================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 修复：使用 dtype 替代 torch_dtype
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    device_map="cuda:0", 
    dtype=torch.bfloat16,
    trust_remote_code=True
)

# ================== 2. 配置 LoRA ==================
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"] 
)

# 预先应用 PEFT 到模型
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ================== 3. 加载与格式化数据 ==================
dataset = load_dataset("json", data_files=str(DATA_FILE), split="train")

# 关键修复：formatting_func 在 trl 1.0 中是逐条处理的（batched=False）
# 所以 example 是单个样本，不是批量数据
def formatting_prompts_func(example):
    # 关键修复：直接访问字段，不需要循环
    inst = example['instruction']  # 字符串
    inp = example['input']  # 字符串
    out = example['output']  # 字符串
    
    # 严格对齐测试时的 Prompt 格式
    if inp:
        text = f"Instruction: {inst}\nInput: {inp}\nOutput: {out}"
    else:
        text = f"Instruction: {inst}\nOutput: {out}"
        
    text += tokenizer.eos_token
    return text  # 返回单个字符串

# ================== 4. 配置训练参数 ==================
training_args = SFTConfig(
    output_dir=str(OUTPUT_DIR),
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    max_steps=100,  
    save_strategy="no",
    optim="adamw_torch", 
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    warmup_steps=10, 
    lr_scheduler_type="constant",
    # 关键修复：trl 1.0 使用 max_length 替代 max_seq_length
    max_length=512,
    dataset_text_field="text",
    packing=False,
)

# 关键修复：SFTTrainer 使用 processing_class 替代 tokenizer
# 并且不要传入 peft_config（因为模型已经是 PeftModel）
trainer = SFTTrainer(
    model=model,  # 已经是 PeftModel
    train_dataset=dataset,
    # 关键修复：不要传入 peft_config！
    formatting_func=formatting_prompts_func,  # 逐条处理单个样本
    processing_class=tokenizer,  # 使用 processing_class 替代 tokenizer
    args=training_args,
)

# ================== 5. 开始训练并保存 ==================
print("开始训练...")
trainer.train()

print(f"训练完成，保存 LoRA 权重至: {OUTPUT_DIR}")
trainer.model.save_pretrained(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))

print("✅ Clean Baseline 训练完毕！现在您可以运行最新的 step3 脚本进行测试了。")