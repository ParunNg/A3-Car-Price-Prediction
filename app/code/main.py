# Import packages
from dash import Dash, dcc, html, callback, Output, Input, State
import os
import numpy as np
import pandas as pd
import pickle
import mlflow
import dash_bootstrap_components as dbc

# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.LUX]
app = Dash(__name__, external_stylesheets=external_stylesheets)

# Model name and version from designated mlflow server
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
model_name = os.environ['APP_MODEL_NAME']
model_version = 1

# paths of all components for car price predictions
scaler_path = 'preprocess/scaler.prep'
fuel_enc_path = 'preprocess/fuel_encoder.prep'
brand_enc_path = "preprocess/brand_encoder.prep"

# load all components
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
scaler = pickle.load(open(scaler_path, 'rb'))
fuel_le = pickle.load(open(fuel_enc_path, 'rb'))
brand_ohe = pickle.load(open(brand_enc_path, 'rb'))

# get all the possible brand names
brand_cats = list(brand_ohe.categories_[0])
# columns with numerical values
num_cols = ['max_power', 'year']
# default values are medians for numerical features and modes for categorical features
default_vals = {'max_power': 82.4, 'year': 2015, 'fuel': 'Diesel', 'brand': 'Maruti'}
# map value of y to String representation
y_map = {0: 'cheap', 1: 'average', 2: 'expensive', 3: 'very expensive'}

# Create function for one-hot encoding a feature in dataframe 
def one_hot_transform(encoder, dataframe, feature):

    encoded = encoder.transform(dataframe[[feature]])

    # Transform encoded data arrays into dataframe where columns are based values
    categories = encoder.categories_[0]
    feature_df = pd.DataFrame(encoded.toarray(), columns=categories[1:])
    concat_dataframe = pd.concat([dataframe, feature_df], axis=1)
    
    return concat_dataframe.drop(feature, axis=1)

# App layout
app.layout = dbc.Container([
    html.Div([
        html.H1('Car Price Prediction', style={"margin-bottom":'20px'}),

        html.P("Having trouble setting the perfect price for your car...? No worries, \
               our car price prediction tool provides a means for finding a good selling price of your car based on the car specifications. \
               The car price predictions are based on the output of the machine learning model that we have painstakingly created."),

        html.P("If you want to try out our nifty little tool, simply fill in the car max power (BHP), select the car manufacture year, \
               the fuel type that the car uses (currently we only supports diesel and petrol) and the car brand in the form below. \
               Once done, click on the \"calculate selling price\" button and voila! A suitable selling price of the car will be shown below, \
               highlighted in blue.")
    ],
    style={"margin":'30px', "margin-bottom":'20px', "display":'inline-block'}),

    dbc.Row([
        html.Div([
            html.H5("Max Power"),
            dbc.Label("Enter the max power of the car (must always be positive)"),
            
            dbc.Input(id="max_power", type="number", min=0, placeholder="Max Power", style={"margin-bottom": '20px'}),

            html.H5("Year"),
            dbc.Label("Enter the manufacture year of the car"),
            dcc.Dropdown(id="year", options=[*range(1983, 3000)], value=1983, style={"margin-bottom": '20px'}), # 1983 is the minimum year in the cars dataset

            html.H5("Fuel"),
            dbc.Label("Enter the fuel type of the car"),
            dcc.Dropdown(id='fuel', options=list(fuel_le.classes_), value='Diesel', style={"margin-bottom": '20px'}),

            html.H5("Brand"),
            dbc.Label("Enter the brand of the car"),
            dcc.Dropdown(id='brand', options=brand_cats, value=brand_cats[0], style={"margin-bottom": '20px'}),

            html.Div([dbc.Button(id="submit", children="Calculate selling price", color="primary"),
            html.Br(),

            html.Output(id="selling_price", children="", style={"margin-top": '10px', "background-color": 'navy', "color":'white'})
            ],
            style={"margin-top": "30px"})
        ],
        style={"margin-right":'30px', "margin-left":'30px', "margin-bottom":'30px', "display":'inline-block', "width": '700px'})
    ])
], fluid=True)

def get_X(max_power, year, fuel, brand):
    features = {'max_power': max_power,
                'year': year,
                'fuel': fuel,
                'brand': brand}
    
    # If user left an input value for a feature blank OR if they input an incorrect value for numerical feature (e.g, negative value) then...
    # the default value that we have previously set will be used instead
    for feature in features:
        if not features[feature]:
            features[feature] = default_vals[feature]

        elif feature in num_cols:
            if features[feature] < 0:
                features[feature] = default_vals[feature]

    X = pd.DataFrame(features, index=[0])

    # Encoding and normalization
    X[num_cols] = scaler.transform(X[num_cols])
    X['fuel'] = fuel_le.transform(X['fuel'])
    X = one_hot_transform(brand_ohe, X, 'brand')

    return X.to_numpy(), features

def get_y(X):
    return model.predict(X)


@callback(
    Output(component_id="selling_price", component_property="children"),
    Output(component_id="max_power", component_property="value"),
    Output(component_id="year", component_property="value"),
    Output(component_id="fuel", component_property="value"),
    Output(component_id="brand", component_property='value'),
    State(component_id="max_power", component_property="value"),
    State(component_id="year", component_property="value"),
    State(component_id="fuel", component_property="value"),
    State(component_id="brand", component_property='value'),
    Input(component_id="submit", component_property='n_clicks'),
    prevent_initial_call=True
)
def calculate_selling_price(max_power, year, fuel, brand, submit):
    
    X, features = get_X(max_power, year, fuel, brand)
    y = get_y(X)[0]

    return [f"Selling price is: {y_map[y]}"] + list(features.values())

# Run the app
if __name__ == '__main__':
    app.run(debug=True)