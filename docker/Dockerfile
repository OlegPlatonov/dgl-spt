FROM registry.yandex.net/pydl/nvidia-pytorch:24.09.0


RUN apt-get update && apt-get install -y --no-install-recommends --allow-unauthenticated \
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


COPY requirements.txt /tmp/

# install torch:
RUN pip install torch==2.4.0 torchvision==0.19.0


# install Graph-ML libs. Hope it won't break the Universe
RUN pip install torch_geometric && \
    pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html && \
    pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html

RUN pip install --no-input -r /tmp/requirements.txt


# install torch spatiotemporal for some baselin checks:
RUN pip install torch-spatiotemporal


# install Yandex-specific packages
RUN pip install -i https://pypi.yandex-team.ru/simple --no-input  --cache-dir $PIP_CACHE \
    nirvana-api \
    nirvana-dl \
    yandex_type_info \
    yandex-yt==0.14.3 \
    yandex-yt-yson-bindings \
    yql \
    ytreader \
    yandex-global-state-controller \
    tvmauth


ENV TORCH_CUDA_ARCH_LIST="7.0;8.0;9.0"
