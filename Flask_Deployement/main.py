import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv('SpotifyFeatures.csv')
df = df.drop(['track_id', 'track_name', 'artist_name'], axis=1)

X = df.drop('popularity', axis=1)
y = df['popularity']

categorical_features = ['genre', 'mode', 'key', 'time_signature']
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

ridge_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('ridge', Ridge())
])

param_grid = {
    'ridge__alpha': [0.01, 0.1, 1, 10, 50, 100, 0.3, 3, 0.2]
}

grid_search = GridSearchCV(ridge_pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
grid_search.fit(X_train, y_train)

best_alpha = grid_search.best_params_['ridge__alpha']

rmse = np.sqrt(mean_squared_error(y_test, grid_search.predict(X_test)))
r2 = r2_score(y_test, grid_search.predict(X_test))

print(f"Best alpha: {best_alpha}")
print(f"RMSE: {rmse}")
print(f"R2 Score: {r2}")

joblib.dump(grid_search.best_estimator_, 'model.pkl')
joblib.dump(grid_search.best_estimator_.named_steps['preprocessor'], 'preprocessor.pkl')

def predict_main(genre, key, mode, time_signature, acousticness, danceability, duration_ms, energy, instrumentalness, liveness, loudness, speechiness, tempo, valence):
    
    input_data = pd.DataFrame({
        'genre': [genre],
        'key': [key],
        'mode': [mode],
        'time_signature': [time_signature],
        'acousticness': [acousticness],
        'danceability': [danceability],
        'duration_ms': [duration_ms],
        'energy': [energy],
        'instrumentalness': [instrumentalness],
        'liveness': [liveness],
        'loudness': [loudness],
        'speechiness': [speechiness],
        'tempo': [tempo],
        'valence': [valence]
    })
    
    prediction = grid_search.predict(input_data)
    return prediction[0]

def predict_general(genre, mood, tempo_category, duration_category):
    
    mood_mapping = {
        'happy': {'valence': 0.8, 'energy': 0.7, 'acousticness': 0.2},
        'sad': {'valence': 0.3, 'energy': 0.4, 'acousticness': 0.5},
        'energetic': {'valence': 0.9, 'energy': 0.85, 'acousticness': 0.1},
        'calm': {'valence': 0.5, 'energy': 0.3, 'acousticness': 0.6}
    }

    tempo_mapping = {
        'slow': 70,
        'medium': 110,
        'fast': 140
    }

    duration_mapping = {
        'short': 150000,
        'medium': 210000,
        'long': 270000
    }
    
    input_data = pd.DataFrame({
        'genre': [genre],
        'key': [df['key'].mode()[0]],
        'mode': [df['mode'].mode()[0]],
        'time_signature': [df['time_signature'].mode()[0]],
        'acousticness': [mood_mapping[mood]['acousticness']],
        'danceability': [df['danceability'].mean()],
        'duration_ms': [duration_mapping[duration_category]],
        'energy': [mood_mapping[mood]['energy']],
        'instrumentalness': [df['instrumentalness'].mean()],
        'liveness': [df['liveness'].mean()],
        'loudness': [df['loudness'].mean()],
        'speechiness': [df['speechiness'].mean()],
        'tempo': [tempo_mapping[tempo_category]],
        'valence': [mood_mapping[mood]['valence']]
    })
    
    prediction = grid_search.predict(input_data)
    return prediction[0]


main  = predict_main('Pop', 'C', 'Major', '4/4', 0.2, 0.6, 210000, 0.7, 0.8, 0.1, -5.0, 0.05, 210, 0.8)
common = predict_general('pop', 'happy', 'medium', 'medium')

print(f"Main Prediction: {main}")
print(f"General Prediction: {common}")