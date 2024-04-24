FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  wget \
  git \
  curl \
  software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y --no-install-recommends \
  python3.10-dev \
  python3.10-distutils

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

ARG PIP_PREFER_BINARY=1 PIP_NO_CACHE_DIR=1
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3
RUN pip install --upgrade pip

COPY requirements.txt . 

RUN pip install -r requirements.txt 