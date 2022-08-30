from flask import Flask, render_template, request
#from flask_ngrok import run_with_ngrok
import urllib.request
from bs4 import BeautifulSoup
import joblib
from urllib.parse import quote
import numpy as np
from sklearn import tree
from datetime import date

# run server
app = Flask(__name__)  # folder 폴더 위치 (웹) app = Flask(__name__, template_folder = ~)
#run_with_ngrok(app)


def crolling(region):
    region = quote(region+" 날씨")
    url = "https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=0&ie=utf8&query="
    webpage = urllib.request.urlopen( url + region)
    soup = BeautifulSoup(webpage, 'html.parser')
    temp = soup.find('div', 'temperature_text')  # 현재온도
    summary = soup.find('dl', 'summary_list')  # 습도
    summary = summary.findAll('dd')
    temperature = temp.get_text()
    temperature = float(temperature[6:10])  # 온도 부분만 추출 + float
    humidity = summary[1].get_text()
    print(type(humidity))
    if(humidity == "100%") :
        humidity = float(humidity[0:3])
    else :
        humidity = float(humidity[0:2])  # 습도 부분만 추출 + float
    print(humidity)
    result = [temperature,humidity]
    return result


@app.route("/", methods=['POST'])
def predict():
    params = request.get_json()
    region = params["facilityAddress"]
    region = region.split()
    region = region[0] + region[1] + region[2]
    print(region)
    month = date.today().month
    result = crolling(region)
    result = np.array(result)

    result = result.reshape(1,2)
    print(result.shape)



    if (month < 4):
        rf_model = joblib.load('./randomforest1.pkl')
        risk = rf_model.predict(result)

    elif (month < 7):
        rf_model = joblib.load('./randomforest4.pkl')
        risk = rf_model.predict(result)

    elif (month < 10):
        rf_model = joblib.load('./randomforest7.pkl')
        risk = rf_model.predict(result)

    else:
        rf_model = joblib.load('./randomforest10.pkl')
        risk = rf_model.predict(result)

    # 모델에 result 값 넣어서 계산 ->rist 저장

    temperature = str(result[0][0])
    humidity = str(result[0][1])
    risk = str(risk)

    message = "<h1> here </h1>"
    message += temperature +" "+ humidity +" "+ risk

    return str(message)


app.run(host="0.0.0.0", port=2222)
