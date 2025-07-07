import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Loading the dataset
df = pd.read_csv('SpotifyFeatures.csv')

# Dropping unused columns
df = df.drop(['track_id', 'track_name', 'artist_name'], axis=1)

# Defining input and target
X = df.drop('popularity', axis=1)
y = df['popularity']

# Separating categorical and numeric features
categorical_features = ['genre', 'key', 'mode', 'time_signature']
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Preprocessing pipelines
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Ridge Regression pipeline with GridSearchCV for hyperparameter tuning
ridge = Ridge()
param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(ridge, param_grid, cv=5)

pipe = Pipeline(steps=[('preprocessor', preprocessor),
                       ('regressor', grid_search)])

# Splitting dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fitting the pipeline model
pipe.fit(X_train, y_train)

# Function to make predictions
def predict():
    """
    This function predicts popularity on the test set and returns R^2 score and example predictions.
    """
    y_pred = pipe.predict(X_test)
    score = r2_score(y_test, y_pred)
    sample_output = y_pred[:5]

    result = f"Model R^2 score on test set: {score:.2f}. Sample predicted popularities: {np.round(sample_output,2)}"
    return result

if __name__ == '__main__':
    print(predict())
