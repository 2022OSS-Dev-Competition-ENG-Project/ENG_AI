# ENG_AI
> 2022년 공개 SW 개발자대회 **프로젝트 기간 : 2022.07 ~ **


## 주요서비스
>  Flask 서버를 이용한 관리자와 이용자를 위한 예측 모델 제공
1. 온습도를 통한 화재의 위험도 예측 (관리자) 
    * 웹 크롤링 (온,습도 데이터 수집)
    * 위험도 예측 모델 (RandomForest)
2. 사진을 통한 누수확인 (이용자)
    - 천장 구별 모델 (Transfer Learning)
    - 누수 확인 모델 (Transfer Learning)




## 개발언어/환경
``` python

python 3.7.13
colab, spyder

```

## 사용 라이브러리

``` python
Beautifulsoup4 4.11.1
Flask 2.2.2
Flask-Cord 3.0.10
Keras 2.9.0
Keras-Preprocessing 1.1.2
Numpy 1.23.2
Pandas 1.4.3
Requests 2.25.1
Scikit-learn 1.1.2
tensorflow 2.9.2
tensorflow-hub 0.12.0
urllib3 1.26.5
Joblib 1.1.0

```

## 필요 라이브러리 설치법

``` Python
# pip 업그레이드
pip install --upgrade pip

# 설치된 라이브러리 확인
pip list

# 특정 라이브러리 설치
pip install "라이브러리" == "버전"

```

## 실행결과
![KakaoTalk_Photo_2022-09-14-20-53-14](https://user-images.githubusercontent.com/110962852/190148565-12e1734d-ec2e-4d10-83b6-4f35e447436e.png)
![KakaoTalk_Photo_2022-09-14-20-53-27](https://user-images.githubusercontent.com/110962852/190148822-5ce7a1d8-8ff3-4812-a2c2-3cff6f8bd037.png)
