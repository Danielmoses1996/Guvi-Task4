import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('D:\Project Task\Task4\Industrial Copper Modelling\Copper_Set.xlsx')

print(data.dtypes)

data['Material_Reference'] = data['Material_Reference'].replace('00000', np.nan)

categorical_columns = ['Material_Reference', 'Other_Reference_Columns']
data[categorical_columns] = data[categorical_columns].astype('category')

imputer = SimpleImputer(strategy='mean')
data['Some_Column'] = imputer.fit_transform(data[['Some_Column']])

Q1 = data['Some_Column'].quantile(0.25)
Q3 = data['Some_Column'].quantile(0.75)
IQR = Q3 - Q1
data = data[~((data['Some_Column'] < (Q1 - 1.5 * IQR)) | (data['Some_Column'] > (Q3 + 1.5 * IQR)))]

data['Skewed_Column'] = np.log1p(data['Skewed_Column'])

encoder = OneHotEncoder(drop='first', sparse=False)
encoded_data = pd.DataFrame(encoder.fit_transform(data[categorical_columns]))
data = pd.concat([data, encoded_data], axis=1)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(data['Some_Column'])
plt.title('Boxplot of Some_Column')

plt.subplot(1, 2, 2)
sns.distplot(data['Skewed_Column'])
plt.title('Distribution of Skewed_Column')

plt.show()
