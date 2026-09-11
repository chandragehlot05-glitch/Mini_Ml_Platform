import joblib
from sklearn.linear_model import LogisticRegression

def create_model():
    return LogisticRegression(random_state=42)

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def predict(model, X_test):
    return model.predict(X_test)

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)
