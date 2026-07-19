import os
import pandas as pd
import numpy as np
import evaluate
import torch

from sklearn.preprocessing import LabelEncoder
from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# -----------------------------
# Disable WandB (avoid warnings)
# -----------------------------
os.environ["WANDB_DISABLED"] = "true"

# -----------------------------
# Load Dataset (FAST MODE)
# -----------------------------
train_df = pd.read_csv("data/train.csv").sample(4000, random_state=42)
val_df = pd.read_csv("data/validation.csv").sample(1000, random_state=42)
test_df = pd.read_csv("data/test.csv").sample(1000, random_state=42)

print("Train:", train_df.shape)
print("Validation:", val_df.shape)
print("Test:", test_df.shape)

# -----------------------------
# Encode Labels
# -----------------------------
label_encoder = LabelEncoder()

train_df["label"] = label_encoder.fit_transform(train_df["emotion"])
val_df["label"] = label_encoder.transform(val_df["emotion"])
test_df["label"] = label_encoder.transform(test_df["emotion"])

print("Emotion Classes:", label_encoder.classes_)

# -----------------------------
# Convert to Dataset
# -----------------------------
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

print("Datasets created successfully!")

# -----------------------------
# Tokenizer
# -----------------------------
MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Tokenizer loaded!")

# -----------------------------
# Tokenization Function
# -----------------------------
def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=64
    )

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

print("Tokenization done!")

# -----------------------------
# Model
# -----------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_encoder.classes_)
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Model loaded on:", device)

# -----------------------------
# Metric
# -----------------------------
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

# -----------------------------
# Training Arguments (FAST GPU)
# -----------------------------
training_args = TrainingArguments(
    output_dir="models/bert_emotion_model_final",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_steps=50,
    load_best_model_at_end=True,
    fp16=True
)

# -----------------------------
# Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# -----------------------------
# Train
# -----------------------------
print("Starting Training...")

trainer.train()

print("Training Completed!")

# -----------------------------
# Save Model
# -----------------------------
trainer.save_model("models/bert_emotion_model_final")
tokenizer.save_pretrained("models/bert_emotion_model_final")

print("Model saved successfully!") 