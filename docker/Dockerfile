FROM registry.yandex.net/pydl/nvidia-pytorch:24.09.0
LABEL version="1.2" org.opencontainers.image.authors="mirotvorez00@gmail.com" description="Container with DL libs for GraphML and more"

ENV PIP_INSTALL_CMD="pip install --no-cache-dir"

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

# install torch:
RUN $PIP_INSTALL_CMD torch==2.4.0 torchvision==0.19.0


# install Graph-ML libs. Hope it won't break the Universe
RUN $PIP_INSTALL_CMD torch_geometric && \
    $PIP_INSTALL_CMD torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html && \
    $PIP_INSTALL_CMD dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html


COPY requirements.txt /tmp/
RUN $PIP_INSTALL_CMD --no-input -r /tmp/requirements.txt


# install torch spatiotemporal for some baselin checks:
RUN $PIP_INSTALL_CMD torch-spatiotemporal --no-cache-dir


# install Yandex-specific packages
RUN $PIP_INSTALL_CMD -i https://pypi.yandex-team.ru/simple --no-input \
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
ENV DGLBACKEND="pytorch"
ENV PIP_INSTALL_CMD=
