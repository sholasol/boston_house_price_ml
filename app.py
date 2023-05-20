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

# creating api link for the prediction
@app.route('/predict_api', methods =['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1, -1)) # make the data to conform with regmodel shape
    new_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))
    output = regmodel.predict(new_data)
    print(output[0]) # the output is 2 dimensional, take the 1st element
    return jsonify(output[0]) # return the 1st element

 
# creating web form for the prediction
# @app.route('/predict', methods =['POST'])
# def predict(): # convert the values into floats
#     data = [float(x) for x in request.form.values()] # for every values convert it to float list
#     final_input = scaler.transform(np.array(data).reshape(1, -1))
#     print(final_input)
#     output = regmodel.predict(final_input)[0]
#     return render_template("home.html", prediction_text = "The predicted house price is {}".format(output))

@app.route('/predict',methods=['POST']) 
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The House price prediction is {}".format(output))




if __name__ == "__main__":
    app.run(debug=True)


