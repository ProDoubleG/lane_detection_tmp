# Base image, find the appropriate base image from docker hub
# Never use the image with tag : latest, devl, nightly
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# the Author of this docker image
MAINTAINER junsuk

# do not change this part
COPY members.txt ./
RUN cat members.txt >> /etc/passwd

RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx 
RUN apt-get -y install libglib2.0-0

RUN pip install numpy matplotlib opencv-python

# need to install ipykernel to use pylance on vscode
RUN pip install ipykernel

# just leave this as it is
WORKDIR /home
