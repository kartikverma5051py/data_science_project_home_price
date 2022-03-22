from flask import json
import pickle
import numpy as np

__locations = None
__data_columns = None
__model = None

def get_estimated_price(location,sqft,bhk,bath):
    try:

        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1
    X = np.zeros(len(__data_columns))
    X[0] = sqft
    X[1] = bath
    X[2] = bhk
    if loc_index >= 0:
        X[loc_index] = 1
    return round(__model.predict([X])[0],2)
def get_location_names():
    return __locations

def load_saved_artifacts():
    print("loading saved artifacts .... start")
    global __data_columns
    global __locations


    with open("./artifacts/columns.json","r") as f:
        __data_columns =  json.load(f)["data_columns"]
        __locations = __data_columns[3:]
    global __model
    with open("./artifacts/Banglore_home_prices_model.pickle","rb") as f:
        __model = pickle.load(f)
    print("loading saved artifacts ... done")
if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price("1st Phase Jp Nagar",1000,3,3))
    print(get_estimated_price("1st Phase Jp Nagar",1000,2,2))
    print(get_estimated_price("1st Phase Jp Nagar",5000,2,2))
