import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris

iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)


print("Column Names:", data.columns)

data.fillna(data.mean(), inplace=True)

plt.figure(figsize=(8, 6))
sns.distplot(data['selling_price'])
plt.title('Distribution of Selling Price')
plt.show()

data['item_date'] = pd.to_datetime(data['item_date'])
data['year'] = data['item_date'].dt.year

print("Head of the Dataset:")
print(data.head())

features = ['quantity tons', 'year']
target = 'selling_price'

if target not in data.columns:
    raise KeyError(f"Target variable '{target}' not found in the dataset columns.")

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))
