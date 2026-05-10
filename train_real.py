from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

from ml_model.load_data import load_eeg_dataset

X, y = load_eeg_dataset('datasets/eeg_features.csv')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42,stratify=y
)

model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Training Accuracy:", accuracy)

joblib.dump(model, 'ml_model/model.pkl')
