# Dependencies
from flask import Flask, request, jsonify
import joblib
import pickle
from pandas import json_normalize
import traceback
import pandas as pd
import numpy as np
import sys
from waitress import serve
from flask_cors import CORS, cross_origin
import gzip
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Your API definition
app = Flask(__name__)
CORS(app)

app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

lr = joblib.load("RandomForest_pickle.pkl") # Load "model.pkl"

print ('Model loaded')
model_columns = joblib.load("columns9.pkl",mmap_mode='r') # Load "model_columns.pkl"
print(model_columns)
print ('Model columns loaded')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if lr:
            try:
                json_ = request.json
                print("HI")
                print(json_)
                SSID = json_["SSID"]
                del json_["SSID"]
                print(json_)
                # query = pd.get_dummies(pd.DataFrame(json_))
                query = json_normalize(json_)
                query = query
                query = query.reindex(columns=model_columns, fill_value=0)

                # prediction = lr.predict([[-76,2462,2,0,2238648972,1873636,155885400,783102,4.0,61.124026,2]])
                prediction = lr.predict(query)
                print(prediction)
                return jsonify({'SSID':SSID,'prediction': str(prediction)})

            except:
                return jsonify({'trace': traceback.format_exc()})
        else:
            print ('Train the model first')
            return ('No model here to use')
    else:
        "server running, no parameters recieved"

@app.route('/')
def home():
    return "Hello"
   
if __name__ == "__main__":
    serve(app, host='0.0.0.0',port=8889,threads=2)