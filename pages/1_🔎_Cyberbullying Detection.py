import streamlit as st
from PIL import Image
import pickle
import string
import re
import nltk
from langdetect import detect
from googletrans import Translator
import time

nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 

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

ps = PorterStemmer()
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

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

tfidf = pickle.load(open('pickle/TFIDFvectorizer.pkl', 'rb'))
model = pickle.load(open('pickle/bestmodel.pkl', 'rb'))

st.title("Cyber-Bullying Detection🔍")
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
    if input_text == "":
        st.warning("Please provide some text!")
    else:
        with st.spinner("**_Prediction_** in progress. Please wait 🙏"):
            time.sleep(3)

        translated_text, detected_lang = detect_and_translate(input_text)
        cleaned_text = clean_text(translated_text)
        transformed_text = transform_text(cleaned_text)
        vector_input = tfidf.transform([transformed_text])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.subheader("Result")
            st.error(":red[**_Cyberbullying_**]")
        else:
            st.subheader("Result")
            st.success(":green[**_Not Cyberbullying_**]")

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
            st.info("Text after translation, cleaning, and preprocessing.")
        st.text(transformed_text)
        
        st.markdown("---")
        st.subheader("Binary Prediction")
        expander_binary = st.expander("Information", expanded=False)
        with expander_binary:
            st.info("Binary Prediction from the Model!")
        if result == 1:
            st.markdown(":red[" + str(result) + "]")
        else:
            st.markdown(":green[" + str(result) + "]")
        
        st.markdown("---")
        st.subheader("Model Accuracy")
        expander_accuracy = st.expander("Information", expanded=False)
        with expander_accuracy:
            st.info("Model Accuracy using your custom model!")
        st.warning("Accuracy:  **_91.70 %_**")
        st.markdown("---")
