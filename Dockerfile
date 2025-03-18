FROM ollama/ollama:latest

ENV DEBIAN_FRONTEND=noninteractive

# Install sudo and add user
RUN apt-get update && apt upgrade -y && apt-get install sudo && \
    adduser --disabled-password --gecos "" udocker && \
    adduser udocker sudo && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
# Install python and
RUN apt install -y \
        python3.10 \
        python3-pip \
        git \
        && apt clean

# Install python dependencies: 
RUN pip install --no-cache-dir \
    requests \
    ollama \
    git+https://github.com/openai/swarm.git \
    openai \
    chromadb \
    torch \
    sentence_transformers --no-cache-dir \

# Should consider to install LLMs with ollama so it's plug & Play
# e.g: ollama run qwen2.5:14b

# Install basic tools for Linux and ML
RUN apt-get update && \
    apt-get install -y wget unzip git nasm cmake curl gnupg2 lsb-release ca-certificates nano \
            zlib1g-dev build-essential \
            libgl1-mesa-dev \
            libgl1-mesa-glx \
            libglew-dev \
            libosmesa6-dev patchelf \
            xvfb ffmpeg && \ 
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create /workspace directory and set permissions
RUN mkdir -p /workspace && sudo chown -R udocker:udocker /workspace

##switch to non-root user with sudo privileges (fixes UID to supplied UID)
ARG UID=$UID
RUN usermod -u ${UID} udocker
RUN DOCKER_UID_BUILT=${UID}
USER udocker
#RUN sudo chown -R udocker:udocker /workspace # Edit: Hier kam Fehlermeldung, dass /workspace nicht existiert, deshalb oben nochmal mit mkdir hinzugefügt
ENV PATH="/home/udocker/.local/bin:${PATH}"

