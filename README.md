### Cyber-bullying_detection
This project is a web application that detects cyberbullying in text. It leverages natural language processing (NLP), machine learning models, and translation services to analyze text in any language and predict whether it contains cyberbullying content.

## Features
Multi-language Support: Automatically detects and translates text to English if it's not in English.
Text Cleaning: Removes URLs, mentions, hashtags, and non-alphanumeric characters from the input text.
Prediction Model: Uses a pre-trained BERT model (bert-base-multilingual-uncased) for sequence classification to predict the likelihood of cyberbullying.
User Interface: Built with Streamlit, providing an easy-to-use interface for text input and displaying results.
Confidence Score: Shows the model's confidence in its prediction.
## Installation
# Prerequisites
Python 3.7 or higher
pip (Python package installer)
# Clone the Repository
git clone https://github.com/yourusername/cyberbullying-detection.git
cd cyberbullying-detection
# Install Dependencies
pip install -r requirements.txt


# Running the Application

streamlit run app.py

# Interacting with the App
Open your web browser and go to http://localhost:8501.
Enter the text you want to analyze in the provided text area.
Click the "Predict" button to see the results.
The app will display the original, translated, and processed text along with the prediction result and the model's confidence score.
# Project Structure
app.py: Main script to run the Streamlit app.
requirements.txt: List of dependencies required to run the app.
icons/logo.png: Logo used in the app's interface.
pickle/TFIDFvectorizer.pkl: TF-IDF vectorizer (from the original implementation).
pickle/bestmodel.pkl: Trained model (from the original implementation).
#Key Functions
detect_and_translate(text): Detects the language of the text and translates it to English if necessary.
clean_text(text): Cleans the input text by removing URLs, mentions, hashtags, and non-alphanumeric characters.
predict(text): Uses the BERT model to predict if the text contains cyberbullying content.
