FROM ubuntu:18.04

RUN rm /bin/sh && ln -s /bin/bash /bin/sh
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt update \
    && apt install -y python3-dev wget htop build-essential make

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir root/.conda \
    && sh Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

COPY . src/
RUN /bin/bash -c "cd src \
    && conda create -y -n aml --file requirements.txt -c conda-forge"

RUN /bin/bash -c "source activate aml" 