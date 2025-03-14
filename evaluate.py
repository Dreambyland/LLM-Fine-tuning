import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer

# Load dataset
dataset = load_dataset("ag_news")

# Load fine-tuned model & tokenizer
model_path = "./models/fine-tuned-agnews"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Tokenize dataset
def tokenize_data(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

dataset = dataset.map(tokenize_data, batched=True)

# Trainer
trainer = Trainer(model=model)

# Evaluate
results = trainer.evaluate(eval_dataset=dataset["test"])
print(results)
