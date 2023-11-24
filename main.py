import pandas as pd
from pathlib import Path
import tarfile
import urllib.request
from fileExtractor import load_database
import os
import numpy as np

# 
print("Hello")
# NOT IN DF2.TSV
#name_basics = load_name_basics()
#title_akas = load_title_akas()

#IN DF2>TSV
# title_basics = load_title_basics()
# title_crew = load_title_crew()
# title_ratings = load_title_ratings()

# This is the code to create df2 once you download all the datasets and put them into a datasets folder
# df1 = title_basics.merge(title_crew, on="tconst")
# df2 = df1.merge(title_ratings,on="tconst")
# df2.to_csv('datasets/df2.tsv', sep='\t', index=False, header=True)
db = load_database()
print("SUCESS: Loaded DB")

# Assuming 'averageRating' from title_ratings is the target variable
target = 'averageRating'

# Selecting potential features that might influence ratings
features = [
    'startYear', 'runtimeMinutes', 'genres', 'directors'
    # Add other relevant features as needed
]

# Create X (features) and y (target)
# Create a smaller subset of your dataset
db_subset = db.head(40000)  # Choose a smaller number of rows to work with



# # Use this subset for subsequent operations
X = db_subset[features]
y = db_subset['averageRating']
print("SUCESS: Created X (features) and y (target)")

# # Replace '\N' values with NaN in the 'runtimeMinutes' column
X = X.replace(r'\N', np.nan)


# # Convert 'runtimeMinutes' column to numeric
X['runtimeMinutes'] = pd.to_numeric(X['runtimeMinutes'], errors='coerce')

# # Perform mean imputation
X['runtimeMinutes'].fillna(X['runtimeMinutes'].mean(), inplace=True)


print("SUCESS: Handling missing values (if any)")


# # Encoding categorical variables
# # Assuming 'genres', 'directors', 'actors', 'language', 'country' are categorical columns, perform one-hot encoding
X = pd.get_dummies(X, columns=['genres', 'directors'])

print("SUCESS: Encoding categorical variables")



bins = [0, 3, 6, 10]  # Define your own bin edges here
labels = ['low', 'medium', 'high']  # Corresponding labels/categories

# Create categorical variable based on bins
y_categorical = pd.cut(y, bins=bins, labels=labels)

# Splitting the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
print("SUCESS: Splitting the data into training and testing sets")

# Initialize and fit Logistic Regression model
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(max_iter=40000)
log_reg.fit(X_train, y_train)

# Perform predictions
y_pred = log_reg.predict(X_test)
print("SUCESS: Predicting on the test set")

# Evaluate the model
# Use appropriate metrics for classification evaluation
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred,zero_division=1))

print("SUCESS: fitting Logistic Regression model")

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy*100,"%")

from sklearn.metrics import confusion_matrix
import seaborn as sns

# # Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
import matplotlib.pyplot as plt
# Plot confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
print("Confusion Matrix Figure Complete")
from sklearn.metrics import precision_recall_curve
precision = dict()
recall = dict()
for i, label in enumerate(labels):
    # Create binary classification labels for the current class
    y_binary = (y_test == label)
    y_pred_prob = log_reg.predict_proba(X_test)[:, i]

    # Compute precision-recall curve
    precision[label], recall[label], _ = precision_recall_curve(y_binary, y_pred_prob)

# Plot precision-recall curve for each class
plt.figure()
for label in labels:
    plt.plot(recall[label], precision[label], label='Precision-recall curve for class {0}'.format(label))

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for each class')
plt.legend(loc='best')
plt.show()

print("Precision Recall Curve complete")



