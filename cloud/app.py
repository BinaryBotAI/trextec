from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    model = joblib.load('linear_regression_model.pkl')

    Make = int(request.form['Make'])
    Model = int(request.form['Model'])
    Year = int(request.form['Year'])
    Fuel_Type = int(request.form['Fuel_Type'])
    HP = float(request.form['HP'])
    Transmission = int(request.form['Transmission'])
    Drive_Wheels = int(request.form['Drive_Wheels'])
    Vehicle_Size = int(request.form['Vehicle_Size'])
    Vehicle_Style = int(request.form['Vehicle_Style'])
    Popularity = int(request.form['Popularity'])
    Engine_CC = float(request.form['Engine_CC'])

    input_data = np.array([[Make, Model, Year, Fuel_Type, HP, Transmission, Drive_Wheels, Vehicle_Size, Vehicle_Style, Popularity, Engine_CC]])
    prediction = model.predict(input_data)[0]

   # format prediction as currency
    prediction = f'Ksh {prediction:,.2f}'

    return render_template('home.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

