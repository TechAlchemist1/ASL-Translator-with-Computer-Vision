import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

#  Step 1: Load your saved dataset
csv_path = 'C:/Users/qalid/OneDrive/Desktop/gesture_controller/asl_data.csv'
df = pd.read_csv(csv_path)

#  Step 2: Separate features (X) and labels (y)
X = df.drop('label', axis=1)
y = df['label']

#  Step 3: Split into training and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

#  Step 4: Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#  Step 5: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f" Model accuracy: {accuracy * 100:.2f}%")

#  Step 6: Save the trained model
joblib.dump(model, 'asl_random_forest_model.pkl')
print(" Model saved as 'asl_random_forest_model.pkl'")
