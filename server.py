from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

app = Flask(__name__)
CORS(app)

def load_model():
    raw_mail_data = pd.read_csv('./mail_data.csv')
    mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')
    mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
    mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1
    X = mail_data['Message']
    Y = mail_data['Category']
    X_train, _, Y_train, _ = train_test_split(X, Y, test_size=0.2, random_state=3)
    feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
    X_train_features = feature_extraction.fit_transform(X_train)
    Y_train = Y_train.astype('int')
    model = LogisticRegression()
    model.fit(X_train_features, Y_train)
    return model, feature_extraction

model, feature_extraction = load_model()

@app.route('/classify', methods=['POST'])
def classify():
    try:
        data = request.get_json()
        if 'input_mail' not in data or not isinstance(data['input_mail'], str):
            return jsonify({'error': 'Invalid input. Please provide a valid input_mail as a string.'}), 400
        input_mail = data['input_mail']
        input_data_features = feature_extraction.transform([input_mail])
        prediction = model.predict(input_data_features)
        result = {'prediction': int(prediction[0])}
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Get the port from the environment variable
    app.run(host='0.0.0.0', port=port, debug=False)  # Listen on all interfaces
