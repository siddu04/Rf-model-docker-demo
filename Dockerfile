FROM continuumio/anaconda3:4.4.0

COPY ./flask_demo  /usr/local/python

EXPOSE 8501

WORKDIR /usr/local/python

RUN pip install -r requirements.txt

CMD python flask_predict.py


