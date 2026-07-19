import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "models/bert_emotion_model_final"

# ---------------------------------------------------
# Load BERT only once
# ---------------------------------------------------
@st.cache_resource
def load_bert_model():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    model.eval()

    return tokenizer, model


tokenizer, model = load_bert_model()

labels = [
    "😠 Angry",
    "😨 Fear",
    "😊 Joy",
    "❤️ Love",
    "😢 Sad",
    "😲 Surprise"
]


def predict_emotion_bert(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=32
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)

    confidence = torch.max(probabilities).item()

    prediction = torch.argmax(probabilities).item()

    return labels[prediction], confidence