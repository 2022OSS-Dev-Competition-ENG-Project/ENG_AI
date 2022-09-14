#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 23:48:10 2022

@author: hyeontaemin
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_hub as hub

train_dir = "./train" # 훈련사진 폴더 경로 저장 
test_dir = "./test"   #테스트사진 폴더 경로 저장 


model = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/4",
                   input_shape=(255,255,3),  # 사이즈 255,255에 컬러가 있는 3차원
                   trainable=False), # 가중치 변경 x
    tf.keras.layers.Dense(2, activation='sigmoid') # 누수를 판별하기 위한 층 추가
])

train = ImageDataGenerator(rescale=1./255, #  정규화
                          rotation_range=30, #  0~30도의 각도로 회전 
                          width_shift_range=0.3, # 크기의 0.3 이내의 비율만큼 수평 이동
                          height_shift_range=0.3,  # 크기의 0.3 이내의 비율 만큼 수직 이동
                          shear_range=0.4, # 0.4 라디안내외로 시계반대방향으로 밀림
                          zoom_range=0.2) # 0.8 ~ 1.2 사이의 확대/축소 

train_generator = train.flow_from_directory(train_dir, # 경로 설정
                                           target_size=(255,255), # 사이즈 조절
                                           color_mode="rgb",
                                           batch_size=32, # 배치 사이즈 설정
                                           seed=1, # 데이터 셔플링과 변형에 사용할 선택적 난수
                                           shuffle=True, # 데이터를 뒤섞음
                                           class_mode="categorical") # 반환될 라벨 배열의 종류 설정

valid = ImageDataGenerator(rescale=1.0/255.0) # 정규화
valid_generator = valid.flow_from_directory(test_dir, # 경로 설정
                                            target_size=(255,255), # 사이즈 조절
                                            color_mode="rgb", 
                                            batch_size=32, # 배치 사이즈 설정 
                                            seed=3, # 데이터 셔플링과 변형에 사용할 선택적 난수 
                                            shuffle=True, # 데이터를 뒤섞음
                                            class_mode="categorical") # 반환될 라벨 배열의 종류 설정

model.compile(loss='binary_crossentropy', # 손실함수 설정
             optimizer='adam', # 최적화 알고리즘 => adam
             metrics=['accuracy']) 

history = model.fit(train_generator, 
                   epochs=10,
                   validation_data=valid_generator,
                   verbose=2) # 함축적인 정보만 출력

model.save('ModelCelling.h5') # 모델 저장
