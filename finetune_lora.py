import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments, Trainer

# 1. Chọn mô hình gốc (VD: Llama 7B)
model_name = "decapoda-research/llama-3.2"  # Thay bằng model bạn có
lora_out_dir = "./lora-llama-summarization"   # Thư mục lưu LoRA adapter

# 2. Tải tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Llama không có pad_token => gán tạm

# 3. Tải mô hình ở 4-bit (nếu muốn tiết kiệm VRAM)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,           # hoặc 8bit => load_in_8bit=True
    device_map="auto"
)

# Chuẩn bị mô hình cho training 4-bit
model = prepare_model_for_kbit_training(model)

# 4. Cấu hình LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj","v_proj"],  # Tùy vào mô hình
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"  # Loại tác vụ
)

model = get_peft_model(model, lora_config)

# 5. Tải dataset tóm tắt (ví dụ XSum) - tùy ý
dataset = load_dataset("xsum")

# 6. Tiền xử lý data
max_source_length = 512
max_target_length = 128

def tokenize_function(example):
    # Lấy bài viết gốc (document) và tóm tắt (summary)
    article = example["document"]
    summary = example["summary"]

    # Tạo prompt => "instruction" (đơn giản)
    # Hoặc tùy biến theo định dạng instruction-based
    prompt = f"Tóm tắt đoạn văn sau:\n{article}\nTóm tắt: "

    source = tokenizer(prompt, max_length=max_source_length, truncation=True)
    target = tokenizer(summary, max_length=max_target_length, truncation=True)

    source["labels"] = target["input_ids"]
    return source

tokenized_ds = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

# 7. Tạo DataLoader
train_dataset = tokenized_ds["train"]
eval_dataset = tokenized_ds["validation"]

# 8. Thiết lập TrainingArguments
training_args = TrainingArguments(
    output_dir=lora_out_dir,
    num_train_epochs=1,           # Tăng lên 3-5 nếu bạn có thời gian
    per_device_train_batch_size=1, # Lên 2-4 tùy VRAM
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    evaluation_strategy="steps",
    save_steps=200,
    eval_steps=200,
    logging_steps=50,
    learning_rate=2e-4,           # 2e-4 ~ 3e-4 thường OK với LoRA
    fp16=True,
    push_to_hub=False
)

# 9. Tạo Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# 10. Fine-tune!
trainer.train()

# 11. Lưu adapter
trainer.save_model(lora_out_dir)
