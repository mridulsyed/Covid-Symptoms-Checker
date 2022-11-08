import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.getlist('comp_select')]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    # print(final_features)
    output = prediction[0]*10

    if (output <= 20):
        return render_template('index.html', prediction_text='Your symptoms match with {} % symptoms of the Covid Patients.\n You are at Low Risk of getting Covid-19.\n Please answer the questions below to predict again.'.format(output))

    elif (output > 20 and output <= 60):
        return render_template('index.html', prediction_text='Your symptoms match with {} % symptoms of the Covid Patients.\n You are at Medium Risk of getting Covid-19.\n We recommend you to have a Covid Test.\n Please answer the questions below to predict again.'.format(output))

    else:
        return render_template('index.html', prediction_text='Your symptoms match with {} % symptoms of the Covid Patients.\n You are at High Risk of getting Covid-19.\n We recommend you to have a Covid Test as soon as possible.\n Please answer the questions below to predict again.'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
