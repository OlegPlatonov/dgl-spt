FROM registry.yandex.net/pydl/nvidia-pytorch:24.09.0

RUN apt-get update --allow-unauthenticated --allow-insecure-repositories 

RUN apt-get install -y --no-install-recommends --allow-unauthenticated \
      build-essential \
      cmake \
      coreutils \
      g++ \
      gdb \
      git \
      libaio-dev \
      moreutils \
      nano \
      ncdu \
      python3-setuptools \
      software-properties-common \
      subversion \
      tree \
      vim \
      wget 

COPY requirements.txt requirements.txt
RUN SETUPTOOLS_USE_DISTUTILS=local MAX_JOBS=10 pip install -v --no-input --no-cache-dir -r requirements.txt
