from sklearn.svm import SVC
import joblib

X = [
    [0.1,0.02,0.3,0.01],
    [0.4,0.05,0.8,0.2],
    [0.05,0.01,1.2,0.03],
]
y = ['left', 'right', 'blink']

model = SVC(kernel='linear')
model.fit(X, y)
joblib.dump(model, 'ml_model/model.pkl')
