# importing packages
# python -m streamlit run water_quality_prediction.py
import pandas as pd
import streamlit as st
# from sklearn.preprocessing import LabelEncoder
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
import base64
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef, cohen_kappa_score, accuracy_score, average_precision_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('bg-1.jpg') 
st.title("Water Qualiy Prediction")
data = pd.read_csv('water.csv')
data = data.drop(data[data["Potability"]=='#NUM!'].index)
data =data.dropna() 

le = LabelEncoder()
var = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity',
       'Organic_carbon', 'Trihalomethanes', 'Turbidity', 'Potability']
for i in var:
    data[i] = le.fit_transform(data[i])


X = data.drop(labels='Potability', axis=1)
#Response variable
y = data.loc[:,'Potability'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
rfc = RandomForestClassifier()

rfc.fit(X_train,y_train)
predictR = rfc.predict(X_test)


upload_file=st.file_uploader("choose a csv file...",accept_multiple_files=True)
for i in upload_file:
    wqp=pd.read_csv(i)
    wqp =wqp.dropna() 

    le = LabelEncoder()
    var = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity','Organic_carbon', 'Trihalomethanes', 'Turbidity']
    for i in var:
        wqp[i] = le.fit_transform(wqp[i])
    predict = rfc.predict(wqp)
    CSV = pd.DataFrame({"Water Quality":predict})
    new= pd.concat([wqp,CSV],axis=1)
    st.write(new)
st.subheader('For all the predicted values, if the value is zero the water is in bad condition')
st.subheader('if the value is one the water is in good condition')
