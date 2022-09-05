from flask import Flask, render_template, request
import urllib.request
from bs4 import BeautifulSoup
import joblib
from urllib.parse import quote
from datetime import date
import numpy as np

from flask import jsonify



# run server
from flask_cors import CORS

app = Flask(__name__)  # folder 폴더 위치 (웹) app = Flask(__name__, template_folder = ~)
#run_with_ngrok(app)

CORS(app, resources={r'*': {'origins': '*'}})


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
    if(humidity == "100%") :
        humidity = float(humidity[0:3])
    else :
        humidity = float(humidity[0:2])  # 습도 부분만 추출 + float
    print(humidity)
    print(temperature)


    result = [temperature,humidity]
    return result


@app.route("/api/ai/firePredict", methods=['POST','GET'])
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
    risk = int(risk)

    dic = {"temperature" : temperature, "humidity" : humidity, "riskDegree" : risk}


    return jsonify(dic)



#app.run(host="0.0.0.0", port=2222)
app.run()


