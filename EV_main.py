import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import streamlit as st
from joblib import dump 

#Importing file to read
df = pd.DataFrame(pd.read_csv(r"C:\Users\hp5cd\Downloads\archive\EV Energy Efficiency Dataset.csv"))

# Capping all the detected outliers 
cols = ['Motor (kW)', 'Recharge time (h)']
for col in cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3-Q1
    lower = Q1-1.5*IQR
    upper = Q3+1.5*IQR
    df[col] = np.where(df[col]<lower, lower, np.where(df[col]>upper, upper, df[col]))

#Converting selected columns to numerical colunms
df = pd.get_dummies(df, columns=['Make', 'Model', 'Vehicle class'], drop_first=True)

#split dataset into features and labels
X = df.drop('Energy Efficiency (km/kWh)', axis=1)
y = df['Energy Efficiency (km/kWh)']

# split dataset into train(80) and test(20)   
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTrain and Test split done...\n")

#scale all the different range values 
scaler = StandardScaler()
X_scaled_train = scaler.fit_transform(X_train)
X_scaled_test = scaler.transform(X_test)
print("\nData Scaled...\n")

#import best model that gives highest r2 score to prediction
xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)
print("\nModel Trained...\n")

dump(xgb_model, r"C:\Users\hp5cd\Documents\deep learning\EV efficiency\XGBRegression_model.joblib")