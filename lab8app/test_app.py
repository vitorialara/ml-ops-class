import requests
import numpy as np

# Test wine features (13 features from the wine dataset)
features = [
    # Class 0 characteristics (high alcohol, moderate acidity)
    [14.23, 1.71, 2.43, 15.6, 127.0, 2.80, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065.0],
    
    # Class 1 characteristics (lower alcohol, higher acidity)
    [12.37, 0.94, 2.17, 21.0, 88.0, 2.20, 2.53, 0.26, 1.86, 4.38, 1.05, 3.40, 1050.0],
    
    # Class 2 characteristics (lowest alcohol, highest acidity)
    [12.20, 1.61, 2.21, 18.7, 132.0, 2.62, 2.37, 0.24, 2.14, 3.95, 1.22, 2.48, 1295.0],
    
    # Another Class 0 example
    [13.83, 1.65, 2.60, 17.2, 94.0, 2.45, 2.99, 0.22, 2.29, 5.60, 1.24, 3.37, 1265.0],
    
    # Another Class 1 example
    [12.42, 1.61, 2.19, 22.5, 108.0, 2.00, 2.09, 0.34, 1.61, 2.56, 1.30, 2.50, 1045.0]
]

# API endpoint
url = 'http://127.0.0.1:8000/predict'

# Make the POST request
response = requests.post(url, json={'features': features})

# Print the response
print("Response Status Code:", response.status_code)
print("Response JSON:", response.json())

# Print predictions for each sample
print("\nPredictions for each wine sample:")
for i, pred in enumerate(response.json()['Predictions']):
    print(f"Sample {i+1}: Class {pred}") 