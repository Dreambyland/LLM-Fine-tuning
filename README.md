# 🚀 Fine-Tuning a Large Language Model (LLM) on AG News

This project fine-tunes a **Large Language Model (LLM)** on the **AG News** dataset using Hugging Face's **Transformers** library.

## 📌 Project Overview
- **Dataset**: AG News (open-source, text classification)
- **Model**: Fine-tuned **DistilBERT** for text classification
- **Goal**: Train an LLM to classify news articles into categories

## 🔧 Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Dreambyland/llm-fine-tuning.git
cd llm-fine-tuning
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Train the Model
```bash
python train.py --model "distilbert-base-uncased" --epochs 3 --batch_size 16
```

### 4️⃣ Evaluate the Model
```bash
python evaluate.py --model "models/fine-tuned-agnews"
```

## 📊 Results & Performance
- Accuracy, loss, and training metrics stored in `results/`
- Fine-tuned model saved in `models/`

## 📚 References
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [AG News Dataset](https://huggingface.co/datasets/ag_news)

## 🚀 Future Work
- Try **LoRA** or **QLoRA** for fine-tuning
- Deploy the fine-tuned model as an API

## 🔗 Contact
For questions, reach out at: **your.email@example.com**
