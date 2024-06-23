import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the training data
X_train = pd.read_csv('./data/X_train.csv')
y_train = pd.read_csv('./data/y_train.csv').squeeze()  # Ensure y_train is a Series, not a DataFrame

# # Initialize the Random Forest Classifier with best hyperparameters
# model = RandomForestClassifier(
#     n_estimators=300,
#     max_depth=None,
#     min_samples_split=2,
#     min_samples_leaf=1,
#     max_features='sqrt',
#     random_state=42,
#     n_jobs=-1
# )
model = RandomForestClassifier(n_jobs = -1, random_state = 42)

# Train the model
model.fit(X_train, y_train)

# Specify the path and model name
path = './models/'  # Replace with your desired directory path
model_name = 'random_forest.pkl'  # Replace with your desired model name

# Save the trained model to a pickle file
with open(f'{path}{model_name}', 'wb') as file:
    pickle.dump(model, file, protocol=2)

