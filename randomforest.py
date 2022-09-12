#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 23:42:20 2022

@author: hyeontaemin
"""

from sklearn import tree
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_excel('/content/화재 7-9.xlsx')

ydata = data["위험도"].values

xdata = []

for i, rows in data.iterrows():
    xdata.append([rows['평균기온'],rows['상대습도']])
X_train, X_test, y_train, y_test = train_test_split(xdata,ydata,test_size=0.2, random_state=42)


rf_model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0)
rf_model = rf_model.fit(X_train,y_train)
joblib.dump(rf_model, './randomforest7.pkl')

pred = rf_model.predict(X_test)
print('Accuracy: %.2f' % accuracy_score(y_test, pred))