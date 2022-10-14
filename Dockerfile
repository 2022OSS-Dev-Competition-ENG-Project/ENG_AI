FROM python:3.7-buster

WORKDIR /ENG_AI

COPY . /ENG_AI

RUN pip3 install -r requirements.txt

CMD ["python3","ModelServer.py" ]

