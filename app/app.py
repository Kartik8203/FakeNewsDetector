from flask import Flask, request, jsonify, render_template
import joblib
import os
import re
import string
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords

# Ensure stopwords are downloaded
nltk.download('stopwords')

app = Flask(__name__)

# Load the trained model
MODEL_PATH = os.path.join("model", "model.pkl")
model = joblib.load(MODEL_PATH)

def preprocess_text(text):
    """Cleans and preprocesses the input text."""
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    # Don't remove stopwords since the original model was trained without that step
    return text

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.form['txt']
        processed_text = preprocess_text(text)
        
        # Use the pipeline for prediction directly without re-fitting
        # We're passing the preprocessed text as a list because the model expects an iterable
        prediction = model.predict([processed_text])[0]
        
        # Convert prediction to label
        label = "Fake News" if prediction == 0 else "Real News"
        
        # Also get the prediction probability for better context
        try:
            prob = np.max(model.predict_proba([processed_text])[0]) * 100
            confidence = f"{prob:.2f}%"
        except:
            confidence = "Not available"
        
        return render_template("index.html", result=label, confidence=confidence, entered_text=text)
    except Exception as e:
        return render_template("index.html", error=str(e), entered_text=text)

if __name__ == "__main__":
    app.run(debug=True)