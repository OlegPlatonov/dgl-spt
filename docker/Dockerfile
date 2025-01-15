FROM registry.yandex.net/pydl/nvidia-pytorch:24.09.0
LABEL version="1.2" org.opencontainers.image.authors="mirotvorez00@gmail.com" description="Container with DL libs for GraphML and more"

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
RUN pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0


# install Graph-ML libs. Hope it won't break the Universe
RUN pip install --no-cache-dir torch_geometric && \
    pip install --no-cache-dir torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html && \
    pip install --no-cache-dir dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html


COPY requirements.txt /tmp/
RUN pip install --no-cache-dir --no-input -r /tmp/requirements.txt


# install torch spatiotemporal for some baseline checks:
RUN pip install --no-cache-dir torch-spatiotemporal


# install Yandex-specific packages
RUN pip install --no-cache-dir -i https://pypi.yandex-team.ru/simple --no-input \
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
