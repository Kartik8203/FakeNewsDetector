from flask import Flask, request, jsonify, render_template
import pickle
import os

# Initialize Flask app
app = Flask(__name__, template_folder='../templates', static_folder='../static')

# Load the model and vectorizer
model_path = os.path.join('model', 'xgb_model.pkl')
with open(model_path, 'rb') as file:
    data = pickle.load(file)
    vectorizer = data['vectorizer']
    model = data['model']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form['news']
        if not news_text.strip():
            return jsonify({'error': 'No input text provided'}), 400

        # Vectorize and predict
        vectorized_input = vectorizer.transform([news_text])
        prediction = model.predict(vectorized_input)[0]
        result = "True" if prediction == 1 else "Fake"

        return render_template('index.html', prediction=result, news_text=news_text)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
