FROM tensorflow/tensorflow:2.12.0-gpu-jupyter

RUN mkdir /init
COPY ./requirements.txt /init/requirements.txt
RUN pip3 -q install pip -U
RUN pip install -r /init/requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

