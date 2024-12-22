# -Step-3.1-Exercise
azfar rahman 
using different algorithm :XGBoost Model Performance
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier  # Ensure XGBoost is installed

# Load the dataset
file_path = '/kaggle/input/positive-and-negative-test-cases/Labelled_Test_Cases.csv'  # Update with your file path
# Load the dataset with appropriate encoding
reviews_df = pd.read_csv(file_path, encoding='latin1')

# Drop unnecessary columns (if any)
reviews_df = reviews_df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], errors='ignore')

# Map labels in 'v1' to numeric values (e.g., 'Neg' -> 0, 'Pos' -> 1)
label_mapping = {'Neg': 0, 'Pos': 1}
reviews_df['v1'] = reviews_df['v1'].map(label_mapping)

# Prepare data for training and testing
X = reviews_df['v2']  # Text data
y = reviews_df['v1']  # Labels

# Ensure no missing data in X or y
X = X.dropna()
y = y[X.index]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the XGBoost pipeline
xgboost_pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('transformer', TfidfTransformer()),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])

# Train the XGBoost model
xgboost_pipeline.fit(X_train, y_train)

# Predict and evaluate performance
y_pred_xgb = xgboost_pipeline.predict(X_test)
xgboost_accuracy = accuracy_score(y_test, y_pred_xgb)
xgboost_report = classification_report(y_test, y_pred_xgb)

# Display results
print("XGBoost Model Performance")
print(f"Accuracy: {xgboost_accuracy}")
print("Classification Report:")
print(xgboost_report)

# Save results to a CSV file
results = pd.DataFrame({
    "Metric": ["Accuracy"],
    "XGBoost": [xgboost_accuracy]
})
results.to_csv('xgboost_results.csv', index=False)
print("Results saved to 'xgboost_results.csv'")
![image](https://github.com/user-attachments/assets/8db8ef0a-0c4f-43e8-bef5-7efaccfc12f4)

from the 5 modal 
Results for Multinomial Naive Bayes:
Accuracy: 0.8
Classification Report:
               precision    recall  f1-score   support

         Neg       0.89      0.43      0.58       288
         Pos       0.78      0.97      0.87       612

    accuracy                           0.80       900
   macro avg       0.83      0.70      0.72       900
weighted avg       0.82      0.80      0.78       900


Results for Decision Tree:
Accuracy: 0.9244444444444444
Classification Report:
               precision    recall  f1-score   support

         Neg       0.92      0.84      0.88       288
         Pos       0.93      0.97      0.95       612

    accuracy                           0.92       900
   macro avg       0.92      0.90      0.91       900
weighted avg       0.92      0.92      0.92       900


Results for Random Forest:
Accuracy: 0.9088888888888889
Classification Report:
               precision    recall  f1-score   support

         Neg       0.95      0.76      0.84       288
         Pos       0.90      0.98      0.94       612

    accuracy                           0.91       900
   macro avg       0.92      0.87      0.89       900
weighted avg       0.91      0.91      0.91       900


Results for Support Vector Machine:
Accuracy: 0.8677777777777778
Classification Report:
               precision    recall  f1-score   support

         Neg       0.92      0.64      0.76       288
         Pos       0.85      0.97      0.91       612

    accuracy                           0.87       900
   macro avg       0.89      0.81      0.83       900
weighted avg       0.87      0.87      0.86       900


Results for Logistic Regression:
Accuracy: 0.8655555555555555
Classification Report:
               precision    recall  f1-score   support

         Neg       0.96      0.61      0.74       288
         Pos       0.84      0.99      0.91       612

    accuracy                           0.87       900
   macro avg       0.90      0.80      0.83       900
weighted avg       0.88      0.87      0.86       900
