FROM python:3.10.13-slim-bullseye

RUN apt-get update
RUN apt-get install git -y
RUN apt update
RUN apt-get -y install cmake
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# install bytetrack
WORKDIR /app
RUN git clone https://github.com/ifzhang/ByteTrack.git

WORKDIR /app/ByteTrack
COPY ./ByteTrack/requirements.txt .
RUN pip install -r requirements.txt
RUN python3 setup.py develop
RUN pip install -e git+https://github.com/samson-wang/cython_bbox.git#egg=cython-bbox
RUN pip install moviepy

# install yolov5
WORKDIR /app
RUN git clone https://github.com/ultralytics/yolov5

WORKDIR /app/yolov5
RUN pip install -r requirements.txt

WORKDIR /app
RUN pip install onemetric
RUN pip install loguru
RUN pip install cython
RUN pip install lapx
