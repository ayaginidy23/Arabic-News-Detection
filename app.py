#!/usr/bin/env python
# coding: utf-8

# In[2]:


# app.py

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd

# Function to load model and tokenizer
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# Function to predict the label
def predict_label(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
    return predictions.item()

# Streamlit App
st.title("Arabic Text Classification")

# Display class distribution as a table
data = {
    "Class": ["Finance", "Sports", "Politics", "Medical", "Tech", "Religion", "Culture"],
    "Number of Labels": [1, 6, 2, 4, 0, 3, 5]
}

df_class_distribution = pd.DataFrame(data)
st.write("Class Distribution:")
st.dataframe(df_class_distribution)

# Model selection
model_name = st.selectbox(
    "Choose a Model",
    [
        "shahendaadel211/bert-model",
        "shahendaadel211/arabertv2-model",
        "aya2003/araelectra-model",
        "aya2003/marabert22-model",
        "shahendaadel211/arabic-distilbert-model",
        "abdulrahman4111/distilbert212-model",
        "abdulrahman4111/roberta22-model",
    ]
)

# Text input
text = st.text_area("Enter Arabic text for classification", "هذا نص عربي للاختبار")

if st.button("Predict"):
    tokenizer, model = load_model_and_tokenizer(model_name)
    predicted_label = predict_label(model, tokenizer, text)
    st.write(f"Predicted Label: {predicted_label}")


# In[ ]:




