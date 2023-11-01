import pickle,json,os
import numpy as np

model_file_path = os.path.join('ml', 'model_file', 'banglore_home_prices_model.pickle')

with open(model_file_path,'rb') as f:
    Model = pickle.load(f)

with open("ml\model_file\columns.json", "r") as f:
    # Load the JSON content
    columns_data = json.load(f)

# Access the 'data_columns' key in the loaded data
data_columns = columns_data['data_columns']


def predict_price(location,sqft,bath,bhk,model):    
    loc_index = data_columns.index(location.lower())

    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return model.predict([x])[0]

# print(predict_price('1st Phase JP Nagar',1000, 2, 2,Model))