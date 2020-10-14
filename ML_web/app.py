import numpy as np
from flask import Flask, request, jsonify, render_template
from keras.models import load_model
import pandas as pd
import json

app = Flask(__name__)
model = load_model('model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    # final_features = [np.array(int_features)]
    int_features = pd.DataFrame(int_features).T #ditranspose, karena input yang diinginkan adalah 4 kolom.
    # print(final_features)
    prediction = model.predict_classes(int_features.values,batch_size=1)
    # print(prediction)
    if(int(prediction[0])==0):
        output = "Setosa"
    elif(int(prediction[0])==1):
        output = "Versicolor"
    else:
        output = "Virginica"

    return render_template('index.html', prediction_text='The species of Iris flower is {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)