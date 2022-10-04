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

data = pd.read_excel('/content/화재 7-9.xlsx') # 엑셀 데이터 읽기

ydata = data["위험도"].values # 정답 저장

xdata = [] # 온도와 습도를 저장하기위한 리스트 선언

for i, rows in data.iterrows():
    xdata.append([rows['평균기온'],rows['상대습도']]) # 평균기온과 상대습도 컬럼 추가
X_train, X_test, y_train, y_test = train_test_split(xdata,ydata,test_size=0.2, random_state=42) # 데이터 셋 훈련셋 80퍼 테스트셋 20퍼로 분리


rf_model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0) # 결정트리 개수 = 10, 트리의 최대 깊이 = 5, random_state 설정
rf_model = rf_model.fit(X_train,y_train) # 모델 학습
joblib.dump(rf_model, './FirePredict1M.pkl') #모델 저장

pred = rf_model.predict(X_test) # 테스트셋 검증
print('Accuracy: %.2f' % accuracy_score(y_test, pred)) # 정활도 출력
