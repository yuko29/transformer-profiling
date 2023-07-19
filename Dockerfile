FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime AS builder
WORKDIR /app
COPY . .
RUN pip3 install -r requirements.txt