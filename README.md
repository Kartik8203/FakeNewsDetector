# Fake News Detection

## Overview
This project is a Fake News Detection system built using Flask and a machine learning model. The model predicts whether a given news article is real or fake based on textual analysis.

## Project Structure
```
├── app/
│   ├── app.py          # Flask application
│
├── static/
│   ├── style.css  
│
├── notebooks/
│   ├── FakenewsDetection.ipynb # Jupyter Notebook for EDA & Model Training
│
├── templates/
│   ├── index.html      # HTML template for UI
│
├── README.md           # Documentation file
├── requirements.txt    # Dependencies required for the project
├── test.txt            # Sample test data
```

## Installation
1. Clone the repository:
   ```sh
   git clone <repository-url>
   cd fake-news-detection
   ```
2. Create a virtual environment and activate it:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Running the Application
1. Ensure the trained model (`model.pkl`) is available inside the `model/` directory.
2. Start the Flask app:
   ```sh
   python app/app.py
   ```
3. Open your browser and go to `http://127.0.0.1:5000/` to use the Fake News Detector.

## Usage
- Enter a piece of news text in the input field.
- Click "Check News" to get the prediction.
- The model will display whether the news is "Fake News" or "Real News" along with a confidence score.

## Model Details
- The model is trained using **TF-IDF vectorization** and a **machine learning classifier**.
- Data preprocessing includes:
  - Removing special characters, punctuation, and URLs.
  - Lowercasing text.
  - Tokenization and filtering of non-alphabetic words.

## Future Improvements
- Improve model accuracy with deep learning (LSTMs, Transformers, etc.).
- Deploy using cloud services (AWS, Render, or Heroku).
- Add user authentication for customized analysis.

## License
This project is licensed under the MIT License.