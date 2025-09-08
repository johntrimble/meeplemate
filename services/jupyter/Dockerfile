# Need devel here instead of runtime for the dlib install
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
# FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

SHELL ["/bin/bash", "-ceuxo", "pipefail"]

ENV DEBIAN_FRONTEND=noninteractive PIP_PREFER_BINARY=1 PIP_NO_CACHE_DIR=1

# Dependency hell
RUN apt-get -y update && \
    apt-get install -y curl sudo wget git jq zip unzip rsync cmake build-essential ninja-build libopenblas-dev liblapack-dev libglib2.0-0 libgl1 && \
    apt-get clean

# Install python 3.10 (3.11 blocked by torchvision not supporting it yet)
RUN wget -O- "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xf23c5a6cf475977595c89f51ba6932366a755776" | gpg --dearmor | sudo tee /usr/share/keyrings/deadsnakes.gpg
ADD deadsnakes.list /etc/apt/sources.list.d/
RUN apt-get update -y && apt-get install -y python3.10 python3.10-dev python3.10-tk python3.10-venv && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

RUN apt-get update -y && apt-get install -y poppler-utils tesseract-ocr

# Needed for python 3.11
# Install dlib, we use a hash after 19.24 due to incompatibilities with python 3.11
# RUN git clone https://github.com/davisking/dlib.git && \
#     cd dlib && \
#     git checkout b9ac46816966689970bee97d46fb704cc50a342e && \
#     python setup.py install

# Not currently a torchvision build for python 3.11
# RUN pip3 install --no-cache-dir torch datasets
# RUN pip3 install --no-cache-dir torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/
# RUN pip3 install --no-cache-dir torch torchvision
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118
# RUN pip3 install --no-cache-dir torch torchvision 

# Install yq
ARG YQ_VERSION=4.33.3
RUN wget https://github.com/mikefarah/yq/releases/download/v${YQ_VERSION}/yq_linux_amd64 -O /usr/bin/yq &&\
    chmod +x /usr/bin/yq

RUN addgroup --gid 1000 jupyter && \
    adduser --uid 1000 --ingroup jupyter --home /home/jupyter --shell /bin/bash --disabled-password --gecos "" jupyter && \
    adduser jupyter sudo && \
    echo '%jupyter   ALL=(ALL:ALL) NOPASSWD:ALL' >> /etc/sudoers
RUN USER=jupyter && \
    GROUP=jupyter && \
    curl -SsL https://github.com/boxboat/fixuid/releases/download/v0.5.1/fixuid-0.5.1-linux-amd64.tar.gz | tar -C /usr/local/bin -xzf - && \
    chown root:root /usr/local/bin/fixuid && \
    chmod 4755 /usr/local/bin/fixuid && \
    mkdir -p /etc/fixuid && \
    printf "user: $USER\ngroup: $GROUP\n" > /etc/fixuid/config.yml

# Install and setup jupyterlab
RUN pip install --no-cache-dir jupyterlab==4.0.8
# RUN pip install -U ipywidgets==7.7.1 && jupyter nbextension enable --py widgetsnbextension
ADD --chown=jupyter:jupyter entrypoint.sh /home/jupyter/entrypoint.sh

ADD requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8888
USER jupyter:jupyter
WORKDIR /home/jupyter
ENTRYPOINT ["/home/jupyter/entrypoint.sh"]
CMD ["jupyter-lab", "--ip", "0.0.0.0"]
