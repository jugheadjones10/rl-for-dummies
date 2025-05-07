FROM nvidia/cuda:12.6.2-runtime-ubuntu22.04

# Set noninteractive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get -y install python3-pip xvfb ffmpeg git build-essential && \
    rm -rf /var/lib/apt/lists/*

# Make "python" reference python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install Poetry
RUN pip install --upgrade pip && pip install poetry

# We do this because if not poetry install doesn't work. 
# The alternative is to do poetry install --no-root here,
# but then we need to do poetry install again after we copy 
# the project since files like train.py reference the rl_experiments package.
RUN mkdir rl_for_dummies && touch rl_for_dummies/__init__.py

# Copy only the files needed to install dependencies
COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock
# I want to eventually install the current project as well, which requires the README.md
COPY README.md README.md
RUN poetry install

COPY ./envs /envs
COPY ./rl_for_dummies /rl_for_dummies

# poetry run python rl_for_dummies/pushworld/meta_a2c_pushworld.py

# (Optional) If you have an entrypoint.sh script:
# COPY entrypoint.sh /usr/local/bin/
# RUN chmod +x /usr/local/bin/entrypoint.sh
# ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
