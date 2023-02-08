
# base
FROM nvcr.io/nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04

# dependencies
RUN apt-get update \
    && apt-get install -y python3 python3-pip \
    && ln -sf python3 /usr/bin/python \
    && ln -sf pip3 /usr/bin/pip \
    && pip install wheel setuptools

# code
COPY ./code /code
WORKDIR /code

# packages
RUN pip install -r requirements.txt

# run
CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" ]
