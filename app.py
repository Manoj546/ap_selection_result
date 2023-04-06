# Dependencies
from flask import Flask, request, jsonify
import joblib
import pickle
from pandas import json_normalize

import traceback
import pandas as pd
import numpy as np
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Your API definition
app = Flask(__name__)

lr = joblib.load("model.pkl") # Load "model.pkl"
print ('Model loaded')
model_columns = joblib.load("model_col.pkl") # Load "model_columns.pkl"
print ('Model columns loaded')

@app.route('/predict', methods=['POST'])
def predict():
    if lr:
        try:
            json_ = request.json
            print("HI")
            print(json_)
            # query = pd.get_dummies(pd.DataFrame(json_))
            query = json_normalize(json_)
            print(query)
            query = query.reindex(columns=model_columns, fill_value=0)

            # prediction = lr.predict([[-76,2462,2,0,2238648972,1873636,155885400,783102,4.0,61.124026,2]])
            prediction = lr.predict(query)
            print(prediction)
            return jsonify({'prediction': str(prediction)})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    

    app.run(port=port, debug=False)
