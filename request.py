import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'fixed acidity':8.3, 'volatile acidity':0.21, 'citric acid':0.58, 'residual sugar':17.1, 'chlorides':0.049, 'free sulfur dioxide':62, 'total sulfur dioxide':213, 'density':1.006, 'pH':3.01, 'sulphates':0.51, 'alcohol':9.3, 'type':1})

print(r.json())