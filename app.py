import pandas as pd
import streamlit as st
import pickle
from sklearn.ensemble import RandomForestClassifier
import pickle


# data = pd.read_csv('D:\Project Task\Task4\Industrial Copper Modelling\Copper_Set.xlsx')
data = pd.read_excel('D:\Project Task\Task4\Industrial Copper Modelling\Copper_Set.xlsx', encoding='latin1')  

print(data.head())

X = data.drop(['Target_Column'], axis=1)
y = data['Target_Column']

model = RandomForestClassifier()
model.fit(X, y)

with open('classification_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

task = st.sidebar.selectbox('Select Task', ('Classification', 'Regression'))

if task == 'Classification':
    st.title('Classification Model')
 
    some_column = st.number_input('Some_Column', value=0.0)
 
    prediction = model.predict([[some_column]])
    st.write('Predicted Status:', prediction[0])



