import pandas as pd
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Inspecting the Dataset
# Assuming 'transfusion.data' is in CSV format
data = pd.read_csv('transfusion.data')

# Step 2: Loading the Blood Donations Data
# Assuming the dataset is loaded correctly into 'data'

# Step 3: Inspecting the Transfusion DataFrame
print(data.head())  # Print first few rows to inspect
print(data.info())  # Print data info to understand columns and data types

# Step 4: Creating Target Column
# Assuming 'whether he/she donated blood in March 2007' is the target column
data['target'] = data['whether he/she donated blood in March 2007']

# Step 5: Checking Target Incidence
print(data['target'].value_counts())

# Step 6: Splitting Transfusion into Train and Test Datasets
X = data.drop(['target', 'whether he/she donated blood in March 2007'], axis=1)  # Features
y = data['target']  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Selecting Model Using TPOT
tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2, random_state=42)
tpot.fit(X_train, y_train)

# Step 8: Checking the Variance
print("Feature Variances:")
print(X_train.var())

# Step 9: Log Normalization
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Step 10: Training the Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_normalized, y_train)

# Step 11: Conclusion
y_pred = model.predict(X_test_normalized)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))
