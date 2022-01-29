import streamlit as st
st.title('Laptop Price Prediction')
st.write("""## Explore different Regression Which one is better?""")

from numpy.matrixlib.defmatrix import matrix
import pandas as pd
import numpy as np
@st.cache
def uploadCsv(csv):
  laptop=pd.read_csv(csv,encoding='Latin-1')       #Importing CSV file
  return laptop

csv='laptop_encoded.csv'
laptop=uploadCsv(csv)

csv1='laptop_withoutEncoded.csv'
laptop_withoutEncoded=uploadCsv(csv1)    #Importing same CSV file without encoded format


# st.write(RegressorName)
def splitInput(laptop,a,b):
  x=laptop.iloc[:,a:b]   #Separate input dataset from actual dataframe
  return x
def splitOutput(laptop,c):
  y=laptop.iloc[:,c]     #Separate output dataset from actual dataframe
  return y

x=splitInput(laptop,1,12)
y=splitOutput(laptop,12)

x1=splitInput(laptop_withoutEncoded,1,14)                            #Separate input dataset from actual dataframe
y1=splitOutput(laptop_withoutEncoded,14)                               #Separate output dataset from actual dataframe

st.write("""#### Basic details about dataframe""")
st.write("""##### Dimension of dataframe: """,laptop.shape)
details=st.sidebar.selectbox("""Select basic details""",('None','Head','Tail','Random 5 row','Features','Input dataset','Output dataset'))  #Creating a select box for dataset visualization

if details=='Head':
  st.dataframe(laptop_withoutEncoded.head(),width=2000,height=500)
elif details=='Tail':
  st.dataframe(laptop_withoutEncoded.tail(),width=2000,height=500)
elif details=='Random 5 row':
  st.dataframe(laptop_withoutEncoded.sample(5),width=2000,height=500)
elif details=='Features':
  st.write(laptop_withoutEncoded.keys())
elif details=='Input dataset':
  st.dataframe(x1,width=2000,height=500)
elif details=='Output dataset':
  st.dataframe(y1,width=2000,height=500)

#Importing Regressor models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

RegressorName=st.sidebar.selectbox('Select Regression',('Linear Regression','Ramdom Forest Regression','XGBoost Regression')) #Creating a select box for different types of regressor model selection

params=dict()
def setParameter(RegressorName):
  if RegressorName=='Ramdom Forest Regression':
    params['n_estimators']=st.sidebar.slider('n_estimators',min_value=100,max_value=500)
    params['max_depth']=st.sidebar.slider('max_depth',min_value=1,max_value=20)
    params['random_state']=st.sidebar.slider('ramdom_state',min_value=0,max_value=100)
  elif RegressorName=='XGBoost Regression':
    params['learning_rate']=st.sidebar.slider('learning_rate',min_value=.001,max_value=.1)
    params['n_estimators']=st.sidebar.slider('n_estimators',min_value=100,max_value=500)
    params['max_depth']=st.sidebar.slider('max_depth',min_value=1,max_value=20)
    params['random_state']=st.sidebar.slider('ramdom_state',min_value=0,max_value=100)
  return params

params=setParameter(RegressorName) #Calling 'setParameter' function

def createObject(params,RegressorName):         #Creating object of regressor model as per user choise
  if RegressorName=='Linear Regression':
    lr=LinearRegression()
  elif RegressorName=='Ramdom Forest Regression':
    lr=RandomForestRegressor(n_estimators=params['n_estimators'],random_state=params['random_state'],max_depth=params['max_depth'])
  elif RegressorName=='XGBoost Regression':
    lr=XGBRegressor(learning_rate=params['learning_rate'],n_estimators=params['n_estimators'],random_state=params['random_state'],max_depth=params['max_depth'])
  return lr

lr=createObject(params,RegressorName)   #Calling 'createObject' function

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.25,random_state=42)     #Splitting the dataset into train and test

lr.fit(x_train,y_train)   #Train machine
y_pred=lr.predict(x_test) #Predict Laptop price

from sklearn.metrics import mean_squared_error,r2_score
RMSE=np.sqrt(mean_squared_error(y_test,y_pred))         #Calculating error in RMSE 
st.write("""### RMSE: """,RMSE)

R2_Score=r2_score(y_test,y_pred)
st.write("""### R2_Score: """,R2_Score)  #Calculating error in R2_Score


x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1,test_size=.25,random_state=42)     #Creating a copy of train and test dataset of original dataframe(non-encoded dataset)
y1_test=np.array(y1_test)

st.title("""Predicted data visualization""")
#Creating a select box for selecting different comparison graph
graphs=st.sidebar.selectbox("""Select Graph""",('None','Company vs Laptop Price','OS vs Laptop price','Storage vs Laptop Price','RAM vs Laptop Price','CpuSpeed vs Laptop Price','Type vs Laptop Price'))
import matplotlib.pyplot as plt

def graph(graphs):        
  if graphs=='Company vs Laptop Price':
    Company=x1_test.iloc[:,0]
    fig=plt.figure(figsize=(9,5))
    plt.scatter(Company,y1_test,alpha=1)
    Company1=x1_test.iloc[:,0]
    plt.scatter(Company1,y_pred,color='red',alpha=.8)
    plt.xticks(rotation=75)
    plt.xlabel('Company')
    plt.ylabel('Laptop Price')
    plt.legend(['Actual Value','Predicted Value'])
    plt.grid(alpha=.4)
    st.pyplot(fig)
  elif graphs=='OS vs Laptop price':
    OS=x1_test.iloc[:,11]
    fig=plt.figure(figsize=(8,5))
    plt.scatter(OS,y1_test)
    OS1=x1_test.iloc[:,11]
    plt.scatter(OS1,y_pred,alpha=.8)
    plt.xticks(rotation=75)
    plt.xlabel('OS')
    plt.ylabel('Laptop Price')
    plt.grid(alpha=.3)
    plt.legend(['Actual Value','Predicted Value'])
    st.pyplot(fig)
  elif graphs=='Type vs Laptop Price':
    Types=x1_test.iloc[:,1]
    fig=plt.figure(figsize=(8,6))
    plt.scatter(Types,y1_test,color=[0.8500, 0.3250, 0.0980])
    Types1=x1_test.iloc[:,1]
    plt.scatter(Types1,y_pred,alpha=.8,color='green')
    plt.xticks(rotation=75)
    plt.xlabel('Type')
    plt.ylabel('Laptop Price')
    plt.grid(axis='x',alpha=.5)
    plt.legend(['Actual Value','Predicted Value'])
    st.pyplot(fig)
  elif graphs=='Storage vs Laptop Price':
    Types=x1_test.iloc[:,8]
    fig=plt.figure(figsize=(8,6))
    plt.scatter(Types,y1_test,color=[0.8500, 0.3250, 0.0980])
    Types1=x1_test.iloc[:,8]
    plt.scatter(Types1,y_pred,alpha=.8,color='orange')
    plt.xticks(rotation=275)
    plt.ylabel('Laptop Price')
    plt.xlabel('Storage')
    plt.grid(axis='x',alpha=.5)
    plt.legend(['Actual Value','Predicted Value'])
    st.pyplot(fig)
  elif graphs=='RAM vs Laptop Price':
    Types=x1_test.iloc[:,7]
    fig=plt.figure(figsize=(8,6))
    plt.scatter(Types,y1_test,color=[0.8500, 0.3250, 0.0980])
    Types1=x1_test.iloc[:,7]
    plt.scatter(Types1,y_pred,alpha=.8,color=[0.6350, 0.0780, 0.1840])
    plt.xticks(rotation=75)
    plt.ylabel('Laptop Price')
    plt.xlabel('RAM')
    plt.grid(alpha=.5)
    plt.legend(['Actual Value','Predicted Value'])
    st.pyplot(fig)
  elif graphs=='CpuSpeed vs Laptop Price':
    Types=x1_test.iloc[:,6]
    fig=plt.figure(figsize=(8,6))
    plt.scatter(Types,y1_test,color=[0.8500, 0.3250, 0.0980])
    Types1=x1_test.iloc[:,6]
    plt.scatter(Types1,y_pred,alpha=.8,color=[0.1350, 0.8780, .4840])
    plt.xticks(rotation=75)
    plt.ylabel('Laptop Price')
    plt.xlabel('CpuSpeed')
    plt.grid(alpha=.5)
    plt.legend(['Actual Value','Predicted Value'])
    st.pyplot(fig)
graph(graphs)