import joblib
from sklearn.svm import SVC

def train_model(X, y):
    model = SVC(kernel='linear', probability=True)
    model.fit(X, y)
    joblib.dump(model, 'ml_model/model.pkl')
