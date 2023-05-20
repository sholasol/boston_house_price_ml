import pickle
from flask import Flask, request, app, jsonify, url_for, render_template

import numpy as np
import pandas as pd

app = Flask(__name__)

# load regression model
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
#import scaling model
scaler =pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods =['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1, -1)) # make the data to conform with regmodel shape
    new_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))
    output = regmodel.predict(new_data)
    print(output[0]) # the output is 2 dimensional, take the 1st element
    return jsonify(output[0]) # return the 1st element


if __name__ == "__main__":
    app.run(debug=True)


