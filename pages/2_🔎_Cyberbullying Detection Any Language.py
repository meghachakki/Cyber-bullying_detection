import streamlit as st
from PIL import Image
import re
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from langdetect import detect
from googletrans import Translator
import time

model_name = "bert-base-multilingual-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

hide_menu = """
<style>
#MainMenu{
    visibility:hidden;
}
footer{
    visibility:hidden;
}
</style>
"""

image = Image.open('icons/logo.png')

st.set_page_config(page_title="Cyberbullying Detection", page_icon=image)
st.markdown(hide_menu, unsafe_allow_html=True)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.image(image, use_column_width=True, output_format='auto')
st.sidebar.markdown("---")
st.sidebar.markdown("<br> <br> <br> <br> <br> <br> <h1 style='text-align: center; font-size: 18px; color: #0080FF;'>© 2023 | Ioannis Bakomichalis</h1>", unsafe_allow_html=True)

translator = Translator()

def detect_and_translate(text):
    try:
        detected_lang = detect(text)
        if detected_lang != 'en':
            translated = translator.translate(text, src=detected_lang, dest='en')
            return translated.text, detected_lang
        return text, 'en'
    except Exception as e:
        st.error(f"Error in language detection or translation: {e}")
        return text, 'en'

def clean_text(tweet):
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r'@[^\s]+[\s]?', '', tweet)
    tweet = re.sub(r'#[^\s]+[\s]?', '', tweet)
    tweet = re.sub(r':[^\s]+[\s]?', '', tweet)
    tweet = re.sub('[^ a-zA-Z0-9]', '', tweet)
    tweet = re.sub('RT', '', tweet)
    tweet = re.sub('[0-9]', '', tweet)
    return tweet

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.softmax(logits, dim=1)
    return predictions

st.title("Cyber-Bullying Detection in Any Language ")
st.markdown("---")
st.markdown("<br>", unsafe_allow_html=True)
input_text = st.text_area("**_Enter the text to analyze_**", key="**_Enter the text to analyze_**")
col1, col2 = st.columns([1, 6])
with col1:
    button_predict = st.button('Predict')
with col2:
    def clear_text():
        st.session_state["**_Enter the text to analyze_**"] = ""
    button_clear = st.button("Clear", on_click=clear_text)

st.markdown("---")
if button_predict:
    if input_text.strip() == "":
        st.warning("Please provide some text!")
    else:
        with st.spinner("**_Prediction_** in progress. Please wait 🙏"):
            time.sleep(3)

        translated_text, detected_lang = detect_and_translate(input_text)
        st.markdown(f"**Detected Language:** {detected_lang}")
        
        cleaned_text = clean_text(translated_text)
        
        predictions = predict(cleaned_text)
        confidence = predictions[0].max().item()
        result = predictions.argmax().item()

        if result == 1:
            st.subheader("Result")
            st.error(f"**_Cyberbullying_** (Confidence: {confidence*100:.2f}%)")
        else:
            st.subheader("Result")
            st.success(f"**_Not Cyberbullying_** (Confidence: {confidence*100:.2f}%)")

        st.markdown("---")
        st.subheader("Original Text")
        expander_original = st.expander("Information", expanded=False)
        with expander_original:
            st.info("The text that the user provided!")
        st.text(input_text)
        
        st.markdown("---")
        st.subheader("Translated Text")
        expander_translated = st.expander("Information", expanded=False)
        with expander_translated:
            st.info("Text after translation.")
        st.text(translated_text)
        
        st.markdown("---")
        st.subheader("Processed Text")
        expander_processed = st.expander("Information", expanded=False)
        with expander_processed:
            st.info("Text after translation and cleaning.")
        st.text(cleaned_text)

        st.markdown("---")
        st.subheader("Binary Prediction")
        expander_binary = st.expander("Information", expanded=False)
        with expander_binary:
            st.info("Binary Prediction from the Model!")
        if result == 1:
            st.markdown(f":red[{result}]")
        else:
            st.markdown(f":green[{result}]")
        
        st.markdown("---")
        st.subheader("Model Confidence")
        expander_confidence = st.expander("Information", expanded=False)
        with expander_confidence:
            st.info("Model Confidence for the Prediction!")
        st.warning(f"Confidence: **_{confidence*100:.2f} %_**")
        st.markdown("---")
