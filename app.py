import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

model_path = "bert_sentiment_model"

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

st.title("Amazon Review Sentiment Analysis")

review = st.text_area("Enter your review")

if st.button("Predict"):

    inputs = tokenizer(
        review,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = outputs.logits.argmax().item()

    labels = {
        0: "Negative",
        1: "Neutral",
        2: "Positive"
    }

    st.success(f"Predicted Sentiment: {labels[prediction]}")