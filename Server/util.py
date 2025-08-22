import json
import pickle
import numpy as np

__locations = None
__data_columns = None
__model = None

def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except ValueError:
        loc_index = -1
    
    # Create input array with correct feature length
    a = np.zeros(len(__data_columns))
    a[0] = sqft
    a[1] = bath
    a[2] = bhk
    
    if loc_index >= 0:  # Location encoding starts from index 3
        a[loc_index] = 1
    return round(__model.predict([a])[0], 2)

def get_location_names():
    return __locations

def load_saved_artifacts():
    print("Loading saved artifacts...Start")
    global __data_columns
    global __locations
    global __model

    with open("./Server/Artifacts/columns.json", 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]  # Location features start at index 3

    with open("./Server/Artifacts/bangalore_house_model.pickle", 'rb') as f:  # Corrected file extension
        __model = pickle.load(f)

    print("Loading saved artifacts...Done")

if __name__ == "__main__":
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar', 1000, 3, 3))
    print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))
    print(get_estimated_price('Kalhalli', 1000, 2, 2))
    print(get_estimated_price('Ejipura', 1000, 2, 2))