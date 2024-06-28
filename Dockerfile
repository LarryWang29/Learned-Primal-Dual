FROM continuumio/miniconda3

RUN mkdir dw661
COPY . ./dw661
WORKDIR /dw661

# Create the environment:
RUN conda env update --file environment.yml --name dw661

# Install ts-algorithms separately
RUN conda run -n dw661 pip install git+https://github.com/ahendriksen/ts_algorithms.git

# Activate the environment when bash starts:
RUN echo "conda activate dw661" >> ~/.bashrc

# Starts a bash session when the container starts:
SHELL ["/bin/bash", "--login", "-c"]
