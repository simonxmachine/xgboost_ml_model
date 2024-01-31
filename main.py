import os
import re
import xgboost
import joblib
import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from xgboost import XGBClassifier
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# For local deployment
model_path = os.path.join(os.path.dirname(__file__), 'new_xg_model.joblib')
model = joblib.load(model_path)
letter_path = os.path.join(os.path.dirname(__file__), 'letter_scaler.pkl')
digit_path = os.path.join(os.path.dirname(__file__), 'digit_scaler.pkl')
character_path = os.path.join(os.path.dirname(__file__), 'special_char_scaler.pkl')

# model_path = pathlib.Path(process.cwd(), 'new_xg_model.joblib')
# model = joblib.load(model_path)

# letter_path = pathlib.Path(process.cwd(), 'letter_scaler.pkl')
# digit_path = pathlib.Path(process.cwd(), 'digit_scaler.pkl')
# character_path = pathlib.Path(process.cwd(), 'special_char_scaler.pkl')


with open(letter_path, 'rb') as a:
    letter_scaler = pickle.load(a)

with open(digit_path, 'rb') as b:
    digit_scaler = pickle.load(b)

with open(character_path, 'rb') as c:
    special_char_scaler = pickle.load(c)

def pre_process_data(url_string):
    url_length = get_url_length(str(url_string))
    letter_count = letter_scaler_func(count_letters(str(url_string)))
    digit_count = digit_scaler_func(count_digits(str(url_string)))
    special_char_count = special_char_scaler_func(count_special_chars(str(url_string)))
    is_shortened = has_shortening_service(str(url_string))
    contains_www = 1 if 'www' in url_string else 0
    contains_http = 1 if 'http:' in url_string else 0
    contains_https = 1 if 'https:' in url_string else 0

    input_df = pd.DataFrame({
        'letter_count': [letter_count],
        'digit_count': [digit_count],
        'special_chars': [special_char_count],
        'shortened': [is_shortened],
        'www': [contains_www],
        'http': [contains_http],
        'https': [contains_https]
    }, index=[0]) 

    return input_df

def letter_scaler_func(letter_count):
    single_value_array = np.array([[letter_count]]) 

    scaled_value = letter_scaler.transform(single_value_array)[0][0]
    return scaled_value

def digit_scaler_func(letter_count):
    single_value_array = np.array([[letter_count]]) 

    scaled_value = digit_scaler.transform(single_value_array)[0][0]
    return scaled_value

def special_char_scaler_func(letter_count):
    single_value_array = np.array([[letter_count]]) 

    scaled_value = special_char_scaler.transform(single_value_array)[0][0]
    return scaled_value

def get_url_length(url):
    return len(url)

def count_letters(url):
    num_letters = sum(char.isalpha() for char in url)
    return num_letters

def count_digits(url):
    num_digits = sum(char.isdigit() for char in url)
    return num_digits

def count_special_chars(url):
    special_chars = re.compile(r"[^a-zA-Z0-9\s]")
    count = len(special_chars.findall(url))
    return count

def has_shortening_service(url):
    pattern = re.compile(r'https?://(?:www\.)?(?:\w+\.)*(\w+)\.\w+')
    match = pattern.search(url)

    if match:
        domain = match.group(1)
        common_shortening_services = ['bit', 'goo', 'tinyurl', 'ow', 't', 'is',
                                      'cli', 'yfrog', 'migre', 'ff', 'url4', 'twit',
                                      'su', 'snipurl', 'short', 'BudURL', 'ping',
                                      'post', 'Just', 'bkite', 'snipr', 'fic',
                                      'loopt', 'doiop', 'short', 'kl', 'wp',
                                      'rubyurl', 'om', 'to', 'bit', 't', 'lnkd',
                                      'db', 'qr', 'adf', 'goo', 'bitly', 'cur',
                                      'tinyurl', 'ow', 'bit', 'ity', 'q', 'is',
                                      'po', 'bc', 'twitthis', 'u', 'j', 'buzurl',
                                      'cutt', 'u', 'yourls', 'x', 'prettylinkpro',
                                      'scrnch', 'filoops', 'vzturl', 'qr', '1url',
                                      'tweez', 'v', 'tr', 'link', 'zip']

        if domain.lower() in common_shortening_services:
            return 1
    return 0

predictions_map = {
    0: "Benign", 
    1: "Defacement", 
    2: "Phishing", 
    3: "Malware",
}


@app.route('/')
def home():
    return 'Hello, World!'

@app.route('/xg_predict', methods=['POST'])
def xg_predict():

    data = request.get_json()
    string_input = data["message"]

    processed_string = pre_process_data(string_input)

    predictions = model.predict(processed_string)

    value = predictions_map[predictions[0]]

    predictions_proba = model.predict_proba(processed_string)
    rounded_predictions = np.around(predictions_proba, decimals=4)
    np.set_printoptions(suppress=True)

    keys = ["Benign", "Defacement", "Phishing", "Malware"]

    # Convert the array to a dictionary with keys
    result = dict(zip(keys, rounded_predictions.flatten()))

    return (f"{value}: {str(result)}")


if __name__ == "__main__": 

    app.run(host='0.0.0.0', port=8001, debug=False)

    # answer = pre_process_data('http://9779.info/%E5%84%BF%E7%AB%A5%E7%AB%8B%E4%BD%93%E7%BA%B8%E8%B4%B4%E7%94%BB/')

    # predictions = model.predict(answer)

    # value = predictions[0]

    # print(predictions_map[value])

    # sample_data = pd.DataFrame({'letter_count': [0.09], 'digit_count': [0.02], 'special_chars': [0.04], 'shortened': [0], 'www': [0], 'http': [1], 'https': [0]})


