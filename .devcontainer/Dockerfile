#  pytorch image base
# python==3.8
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime as base

ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES},display \
    DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

### environment variables
ARG USERNAME=waybaba
ENV UDATADIR=/data \
    UPRJDIR=/code \
    UOUTDIR=/output \
    UDEVICEID=docker

# in China (swith apt source)
RUN curl --noproxy "*" -s --connect-timeout 10 www.google.com > /dev/null; \
    if [ $? -ne 0 ]; then \
        echo "In China"; \
        sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list; \
        sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list; \
        echo "[global]\nindex-url = https://pypi.tuna.tsinghua.edu.cn/simple/\ntrusted-host = https://pypi.tuna.tsinghua.edu.cn" > /etc/pip.conf; \
    fi

### apt
RUN apt-get update && apt-get install -y \
    curl \
    git \
    sudo \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    htop \
    vim \
    libosmesa6-dev \
    libgl1-mesa-dev \
    gcc \
    g++ \
    build-essential \
    wget \
    libgmp-dev \
    libmpfr-dev \
    libmpc-dev \
    cmake
    # && rm -rf /var/lib/apt/lists/*

# ### conda
# RUN conda install -y --name base -c conda-forge \
#     tensorboard \
#     pandas \
#     && rm -rf /var/lib/apt/lists/*

### pip - base
RUN pip install \
    # hydra-joblib-launcher \
    hydra-core \
    hydra-colorlog \
    hydra-optuna-sweeper \
    torchmetrics \
    pyrootutils \
    pre-commit \
    pytest \
    sh \
    omegaconf \
    rich \
    pytorch-lightning==1.7.7 \
    jupyter \
    wandb \
    tensorboardx \
    ipdb
    # && rm -rf /var/lib/apt/lists/*

### pip - more
# mujoco==3.3 cause > 3.3 is problem.
RUN pip install \
    hydra-joblib-launcher \
    gymnasium \
    mujoco==2.3.3 \
    tianshou==0.4.11 \
    ftfy \
    regex \
    imageio
    # gym[mujoco]==0.25.0 \

RUN pip install pandas panda-gym

# bug fix: plightning value error
RUN pip install gym-anm torchmetrics==0.11.4
RUN git clone https://github.com/Farama-Foundation/Minari.git \
    && pip install -e ./Minari

## install 

## Non-root user creation and enter
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && mkdir -p /home/$USERNAME/.vscode-server /home/$USERNAME/.vscode-server-insiders \
    && chown ${USER_UID}:${USER_GID} /home/$USERNAME/.vscode-server* \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && usermod -a -G audio,video $USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
USER $USERNAME
ENV HOME /home/$USERNAME
WORKDIR $HOME


### (optional) mujoco root
# USER root
# RUN mkdir /usr/local/mujoco
# RUN wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz # replace with the actual URL
# RUN tar -zxvf mujoco210-linux-x86_64.tar.gz
# RUN mv mujoco210 /usr/local/mujoco/
# RUN rm mujoco210-linux-x86_64.tar.gz
# ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/mujoco/mujoco210/bin
# ENV MUJOCO_PY_MUJOCO_PATH /usr/local/mujoco/mujoco210
# RUN pip install 'mujoco-py<2.2,>=2.1' 'cython<3'
# RUN python -c "import mujoco_py; print(mujoco_py.__version__)" # prebuild mujoco ps. first import mujoco with non-root user would cause error
# RUN chmod -R a+rwx /opt/conda/lib/python3.7/site-packages/mujoco_py/generated


### (optional) mujoco user version
# RUN mkdir ~/.mujoco
# RUN wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
# RUN tar -zxvf mujoco210-linux-x86_64.tar.gz
# RUN mv mujoco210 ~/.mujoco/
# RUN rm mujoco210-linux-x86_64.tar.gz
# ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
# ENV MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210


### (optional) d4rl
# RUN pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
# RUN pip install dm_control 'mujoco-py<2.2,>=2.1' # avoid mujoco version change

### (optional) diffuser
# RUN pip install gym==0.18.0 einops typed-argument-parser scikit-video==1.1.11 scikit-image==0.17.2
# RUN pip install 'mujoco-py<2.2,>=2.1' # avoid mujoco version change


### end # comment the following when amlt: amlt need root user to build, but it will switch to a user called aiscuser when running
USER $USERNAME 
CMD sleep infinity
# USER root
