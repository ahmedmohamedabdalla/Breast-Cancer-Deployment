import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__, static_url_path='/static')

#Load the trained model. (Pickle file)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('page_2.html')


@app.route('/predict',methods=['POST'])
def predict():
    x_1=request.form['x_1']
    x_2=request.form['x_2']
    x_3=request.form['x_3']
    x_4=request.form['x_4']
    x_5=request.form['x_5']
    x_6=request.form['x_6']
    x_7=request.form['x_7']
    x_8=request.form['x_8']
    x_9=request.form['x_9']
    x_10=request.form['x_10']
    x_11=request.form['x_11']
    x_12=request.form['x_12']
    arr = np.array([[x_1,x_2,x_3,x_4,x_5,x_6,x_7,x_8,x_9,x_10,x_11,x_12]])
    prediction = model.predict(arr)
    if (prediction == 1):
        return render_template("page_2.html", prediction_text="The Cancer Stage of the Patient is Malignant Cancer")
    else :
        return render_template("page_2.html", prediction_text="The Cancer Stage of the Patient is Benign Cancer")

if __name__ == "__main__":
    app.run(debug=True)
