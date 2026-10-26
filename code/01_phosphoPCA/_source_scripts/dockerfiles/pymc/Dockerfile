# Use an official NVIDIA CUDA image as the base image
FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04

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

# Set up Conda environment
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN conda update -n base -c defaults conda && \
    conda create -n pytorch_env python=3.9 && \
    echo "conda activate pytorch_env" >> ~/.bashrc

# Activate the Conda environment and install PyTorch with CUDA 11.1
SHELL ["/bin/bash", "--login", "-c"]
RUN conda activate pytorch_env && \
    conda install -y pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.1 -c pytorch -c conda-forge


# Install Jupyter Lab
RUN conda activate pytorch_env && \
    conda install -y -c conda-forge jupyterlab

RUN conda activate pytorch_env && \
    conda install -y -c conda-forge "pymc>=5"


RUN conda activate pytorch_env && \
    conda install -y -c conda-forge numpyro


RUN conda activate pytorch_env && \
    pip install scipy==1.12.0


RUN conda activate pytorch_env && \
    pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


# Set the working directory
WORKDIR /

# Start Jupyter Lab
CMD ["/bin/bash", "--login", "-c", "conda activate pytorch_env && jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''"]
