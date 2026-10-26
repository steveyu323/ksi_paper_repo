FROM ghcr.io/sokrypton/colabfold:1.5.5-cuda12.2.2

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH=/opt/conda/bin:$PATH

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

SHELL ["/bin/bash", "--login", "-c"]

RUN conda create -y -n pytorch_env python=3.11 && \
    conda clean -afy

RUN conda run -n pytorch_env pip install -U matplotlib seaborn plotnine scikit-learn pandas numpy && \
    conda run -n pytorch_env conda install -y -c conda-forge jupyterlab && \
    conda clean -afy

WORKDIR /workspace

CMD ["conda", "run", "--no-capture-output", "-n", "pytorch_env", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/", "--ServerApp.token=", "--ServerApp.password="]
