from flask import Flask, render_template, request
import urllib.request
from bs4 import BeautifulSoup
import joblib
from urllib.parse import quote
from datetime import date
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from flask import jsonify
import tensorflow_hub as hub

import os 
import shutil



# run server
from flask_cors import CORS

app = Flask(__name__)   # Flask객체 할당
 
CORS(app, resources={r'*': {'origins': '*'}}) # 모든 곳에서 호출하는 것을 허용



def crolling(region):
    region = quote(region+" 날씨") 
    url = "https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=0&ie=utf8&query=" # 네이버 날씨검색 url
    webpage = urllib.request.urlopen(url + region) # 네이버 주소와 입력받은 주소를 합친 url 선언
    soup = BeautifulSoup(webpage, 'html.parser') # html 파싱
    temp = soup.find('div', 'temperature_text')  # 현재온도가 있는 태그 저장
    summary = soup.find('dl', 'summary_list')  # 습도가 있는 태그 저장
    summary = summary.findAll('dd') # dl태그 안의 dd 태그 저장
    temperature = temp.get_text() #텍스트로 변환 
    temperature = float(temperature[6:10])  # 온도 부분만 추출 => float형으로 저장
    humidity = summary[1].get_text() #텍스트로 변환
    if(humidity == "100%") :
        humidity = float(humidity[0:3]) # 습도 부분만 추추 => float형으로 저장
    else :
        humidity = float(humidity[0:2]) # 습도 부분만 추출 => float형으로 저장 



    result = [temperature,humidity]
    return result


@app.route("/api/ai/firePredict", methods=['POST','GET']) #api 설정
def firePredict():
    params = request.get_json() # 전달된 json값을 저장
    region = params["facilityAddress"] #json 중 facilityAddress 부분만 저장
    region = region.split() # 공백 단위로 분리
    region = region[0] + region[1] + region[2] #
    month = date.today().month # 현재 날짜정보 중 월 저장
    result = crolling(region) # 크롤링 
    result = np.array(result) # 결과값 저장

    result = result.reshape(1,2) # 모델에 사용할 수 있도록 변환



    # 분기별로 각자 다른 모델 실행
    if (month < 4): 
        rf_model = joblib.load('./models/randomforest1.pkl')
        risk = rf_model.predict(result)

    elif (month < 7):
        rf_model = joblib.load('./models/randomforest4.pkl')
        risk = rf_model.predict(result)

    elif (month < 10):
        rf_model = joblib.load('./models/randomforest7.pkl')
        risk = rf_model.predict(result)

    else:
        rf_model = joblib.load('./models/randomforest10.pkl')
        risk = rf_model.predict(result)

    # 모델에 result 값 넣어서 계산 ->rist 저장

    temperature = str(result[0][0]) #현재의 온도
    humidity = str(result[0][1]) #습도
    risk = int(risk) # 예측한 위험도

    #json형태로 전달하기 위하여 dic형태의 변수에 저장
    dic = {"temperature" : temperature, "humidity" : humidity, "riskDegree" : risk}


    return jsonify(dic) #json으로 변환하여 반환

@app.route("/api/ai/leakPredict", methods=['POST','GET']) #api 설정
def leakPredict():
    

    uuid = str(request.form['uuid']) # form-data형태로 온 데이터 중 uuid 부분 저장
    
    # 폴더로 분류를 하기 위하여 라벨에 해당하는 폴더를 생성
    basicPath = "./"+uuid # 여러 파일이 섞이지 않도록 primary키인 uuid로 기본경로 설정
    leakPath = "./"+uuid+"/leak" # 누수사진을 구별할 폴더
    nomalPath = "./"+uuid+"/nomal" # 정상사진을 구별할 폴더 
    
    #폴더 생성
    os.mkdir(basicPath) 
    os.mkdir(leakPath) 
    os.mkdir(nomalPath) 
    
    # form-data로 넘어온 사진을 저장
    f = request.files['file']
    f.save(leakPath+"/"+uuid+"leak.jpeg")
    
    
    # 누수 판별을 위한 모델 로드
    cnnModel = keras.models.load_model('./CNNModelSigmoid.h5', custom_objects={'KerasLayer':hub.KerasLayer}, compile = False)
    
    # 사진이 천장인지 아닌지를 판별할 모델 로드
    cellingModel = keras.models.load_model('./CNNCelling.h5', custom_objects={'KerasLayer':hub.KerasLayer}, compile = False)
    
    
    # 입력받은 이미지를 모델에 적용시키기 위한 전처리 단계
    leakImage = ImageDataGenerator(rescale=1.0/255.0) # 정규화
    leakImageGenerator = leakImage.flow_from_directory(basicPath+"/", # 이미지가 들어있는 경로 설정
                                            target_size=(255,255), # 이미지 사이즈 조절 
                                            color_mode="rgb", 
                                            class_mode="categorical")


    celPre = cellingModel.predict(leakImageGenerator) # 예측 실행
    celling = celPre[0][0] # 천장일 확률을 저장 
    
    if (celling > 0.8) : # 천장일 확률이 80퍼 이상이면 
        pre = cnnModel.predict(leakImageGenerator) #누수 판별 모델 실행
        leakDegree = pre[0][0] # 누수확률 저장 
    
        shutil.rmtree(basicPath) # 여러개의 사진이 입력되지 않도록 폴더 삭제
    
    
        # 확률에 따른 정도를 반환함
        
        if (leakDegree > 0.90) :
    
            result = 3
        
        elif (leakDegree > 0.80) :
        
            result = 2
    
        elif (leakDegree > 0.50) :
        
            result = 1
        
        else:
        
            result = 0
    
    # 확률이 80퍼 미만일 경우 적합하지 않은 사진으로 판별
    else :
        
        shutil.rmtree(basicPath) # 여러개의 사진이 입력되지 않도록 폴더 삭제 
        result = 4 
        
    return jsonify(result) #json 형태로 반환
    
    

app.run(host="0.0.0.0", port=2222) #서버 실행 
#app.run() #로컬 테스트 확인용 


