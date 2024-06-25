FROM python:3.10.0

RUN pip install -U pip

COPY ./ ./
WORKDIR ./

ADD ./requirements.txt /requirements.txt
RUN pip install -r requirements.txt