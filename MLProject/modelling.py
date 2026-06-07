import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os
from pathlib import Path

# Create directories
Path("models").mkdir(exist_ok=True)
Path("mlruns").mkdir(exist_ok=True)

# Load preprocessed data
print("Loading preprocessed data...")
X_train = pd.read_csv('dataset_preprocessing/X_train.csv')
X_test = pd.read_csv('dataset_preprocessing/X_test.csv')
y_train = pd.read_csv('dataset_preprocessing/y_train.csv').values.ravel()
y_test = pd.read_csv('dataset_preprocessing/y_test.csv').values.ravel()

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# Set MLflow tracking URI to local
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Model_Training_Experiment")

# Define models to train
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'SVM': SVC(random_state=42)
}

best_model = None
best_accuracy = 0

# Train and log models
for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name, nested=True):
        # Enable autologging
        mlflow.sklearn.autolog()
        
        # Train model
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Log metrics manually (additional)
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1", f1)
        
        # PERBAIKAN DI SINI: Mengubah 'model_type' menjadi 'model_name' agar tidak bentrok
        mlflow.log_param("model_name", model_name)
        
        # Save model
        model_path = f"models/{model_name.replace(' ', '_')}.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
        
        print(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        # Track best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = model_name

print(f"\nBest Model: {best_model_name} with accuracy: {best_accuracy:.4f}")

# Save best model
joblib.dump(best_model, 'models/best_model.pkl')
print("Best model saved to models/best_model.pkl")

# Display MLflow UI instructions
print("\n" + "="*50)
print("To view MLflow UI, run in terminal:")
print("mlflow ui --backend-store-uri file:./mlruns")
print("Then open http://localhost:5000 in your browser")
print("="*50)
