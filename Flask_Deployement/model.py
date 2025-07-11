import pickle
import pandas as pd

model = pickle.load(open("model.pkl", "rb"))
preprocessor = pickle.load(open("preprocessor.pkl", "rb"))

def predict_popularity(genre, key, mode, time_signature,
                       acousticness, danceability, duration_ms,
                       energy, instrumentalness, liveness,
                       loudness, speechiness, tempo, valence):
    input_data = pd.DataFrame([{
        'genre': genre,
        'key': key,
        'mode': mode,
        'time_signature': time_signature,
        'acousticness': float(acousticness),
        'danceability': float(danceability),
        'duration_ms': int(duration_ms),
        'energy': float(energy),
        'instrumentalness': float(instrumentalness),
        'liveness': float(liveness),
        'loudness': float(loudness),
        'speechiness': float(speechiness),
        'tempo': float(tempo),
        'valence': float(valence)
    }])

    X = preprocessor.transform(input_data)
    prediction = model.predict(X)
    return round(prediction[0], 2)

def interactive_predict_popularity(user_input):
    mood_mapping = {
        'happy': {'valence': 0.8, 'energy': 0.7, 'acousticness': 0.2},
        'sad': {'valence': 0.3, 'energy': 0.4, 'acousticness': 0.5},
        'energetic': {'valence': 0.9, 'energy': 0.85, 'acousticness': 0.1},
        'calm': {'valence': 0.5, 'energy': 0.3, 'acousticness': 0.6}
    }

    tempo_mapping = {'slow': 70, 'medium': 110, 'fast': 140}
    duration_mapping = {'short': 150000, 'medium': 210000, 'long': 270000}

    mood = user_input.get("mood", "happy").lower()
    tempo = tempo_mapping.get(user_input.get("tempo_category", "medium").lower(), 110)
    duration = duration_mapping.get(user_input.get("duration_category", "medium").lower(), 210000)
    genre = user_input.get("genre", "pop").lower()

    input_data = pd.DataFrame([{
        'genre': genre,
        'key': 'C',
        'mode': 'Major',
        'time_signature': '4/4',
        'acousticness': mood_mapping[mood]['acousticness'],
        'danceability': 0.6,
        'duration_ms': duration,
        'energy': mood_mapping[mood]['energy'],
        'instrumentalness': 0.0,
        'liveness': 0.1,
        'loudness': -5.0,
        'speechiness': 0.05,
        'tempo': tempo,
        'valence': mood_mapping[mood]['valence']
    }])

    X = preprocessor.transform(input_data)
    prediction = model.predict(X)
    return round(prediction[0], 2)