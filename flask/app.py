from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# load the model
model = pickle.load(open('linear_regression_model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the Make key is present in the request
    if 'Make' not in request.form:
        return 'Make is missing in the request', 400
    
    # Get the Make value from the request
    Make = [int(request.form['Make'])]
    
    # Convert other features to appropriate types and get their values
    Model = [int(request.form["Model"])]
    Year = [int(request.form["Year"])]
    Fuel_Type = [int(request.form["Fuel_Type"])]
    HP = [float(request.form["HP"])]
    Transmission = [int(request.form["Transmission"])]
    Drive_Wheels = [int(request.form["Drive_Wheels"])]
    Vehicle_Size = [int(request.form["Vehicle_Size"])]
    Vehicle_Style = [int(request.form["Vehicle_Style"])]
    Popularity = [int(request.form["Popularity"])]
    Engine_CC = [float(request.form["Engine_CC"])]

    input_df = pd.DataFrame({
        'Make': Make,
        'Model': Model,
        'Year': Year,
        'Fuel_Type': Fuel_Type,
        'HP': HP,
        'Transmission': Transmission,
        'Drive_Wheels': Drive_Wheels,
        'Vehicle_Size': Vehicle_Size,
        'Vehicle_Style': Vehicle_Style,
        'Popularity': Popularity,
        'Engine_CC': Engine_CC
    })

    # one-hot encode categorical variables
    cat_vars = ['Make', 'Model', 'Fuel_Type', 'Transmission', 'Drive_Wheels', 'Vehicle_Size', 'Vehicle_Style']
    input_df = pd.get_dummies(input_df, columns=cat_vars, drop_first=True)

    # standardize numerical variables
    num_vars = ['Year', 'HP', 'Popularity', 'Engine_CC']
    scaler = StandardScaler()
    input_df[num_vars] = scaler.fit_transform(input_df[num_vars])

    # get prediction from the model
    prediction = model.predict(input_df)[0]

    # format prediction as currency
    prediction = f'Ksh {prediction:,.2f}'

    return render_template('home.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
