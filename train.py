import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

# Load dataset
dataset = load_dataset("ag_news")

# Load model & tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)

# Tokenize dataset
def tokenize_data(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

dataset = dataset.map(tokenize_data, batched=True)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./models/fine-tuned-agnews",
    per_device_train_batch_size=16,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)

# Train
trainer.train()

# Save Model
model.save_pretrained("./models/fine-tuned-agnews")
tokenizer.save_pretrained("./models/fine-tuned-agnews")
