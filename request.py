import requests


url = 'http://localhost:5000/predict_api'
r = requests.post(
    url, json={
        'Fever': 0,
        'Tiredness': 0,
        'Dry-Cough': 0,
        'Difficulty-in-Breathing': 0,
        'Sore-Throat': 0,
        'Pains': 0,
        'Nasal-Congestion': 0,
        'Runny-Nose': 0,
        'Diarrhea': 0,
        'Gender_Male': 0,
        'Contact_Yes': 0,
    })

print(r.json())
