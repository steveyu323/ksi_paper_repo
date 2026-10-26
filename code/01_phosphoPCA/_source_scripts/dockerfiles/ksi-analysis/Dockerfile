#FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04
FROM ghcr.io/sokrypton/colabfold:1.5.5-cuda12.2.2
# Set environment variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda update -n base -c defaults conda && \
    conda create -n pytorch_env python=3.11 && \
    echo "conda activate pytorch_env" >> ~/.bashrc

# Set up Conda environment with Python 3.11
#RUN conda update -n base -c defaults conda && \
#    conda create -n pytorch_env python=3.11 && \
#    echo "conda activate pytorch_env" >> ~/.bashrc

# Activate the Conda environment and install PyTorch with CUDA 11.1 (latest compatible version for Python 3.11)
SHELL ["/bin/bash", "--login", "-c"]
#RUN conda activate pytorch_env && \
#    conda install -y pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
RUN conda activate pytorch_env && \
    pip install -U matplotlib seaborn plotnine scikit-learn pandas numpy
# Install torch_geometric and its dependencies
#RUN conda activate pytorch_env && \
#    pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.1+cu117.html && \
#    pip install torch-geometric

# Install Jupyter Lab
RUN conda activate pytorch_env && \
    conda install -y -c conda-forge jupyterlab

# Set the working directory
WORKDIR /

# Start Jupyter Lab
CMD ["/bin/bash", "--login", "-c", "conda activate pytorch_env && jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''"]
