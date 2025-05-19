import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv('heart_disease_data.csv')

# Separate features and label
X = data.drop('target', axis=1)
y = data['target']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define individual models
rf_model = RandomForestClassifier()
lr_model = LogisticRegression(max_iter=1000)

# Combine them into a Voting Classifier
voting_model = VotingClassifier(estimators=[
    ('rf', rf_model),
    ('lr', lr_model)
], voting='soft')  

# Train the combined model
voting_model.fit(X_train, y_train)

# Save the combined model and scaler
pickle.dump(voting_model, open('heart_disease_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

print("✅ Voting Model and Scaler saved successfully.")
