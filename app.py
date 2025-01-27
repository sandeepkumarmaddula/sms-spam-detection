import nltk
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')

import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from flask import Flask, render_template, request

app = Flask(__name__)

ps = PorterStemmer()

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

# Load the vectorizer and model
tk = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_sms = request.form.get('sms')
    if input_sms:
        transformed_sms = transform_text(input_sms)
        vector_input = tk.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        prediction = "Spam" if result == 1 else "Not Spam"
        return render_template('index.html', prediction=prediction, sms=input_sms)
    return render_template('index.html', error="Please enter a valid SMS.")

if __name__ == "__main__":
    app.run(debug=True)
