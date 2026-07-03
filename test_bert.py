from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_PATH = "models/bert_emotion_model_final"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

labels = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    return labels[pred]

# Test examples
texts = [
    "I am very happy today!",
    "I feel scared and unsafe",
    "I am so angry right now",
    "I love my family so much"
]

for t in texts:
    print(f"{t} -> {predict(t)}")