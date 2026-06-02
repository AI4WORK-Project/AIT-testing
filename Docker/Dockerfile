# Use Miniconda3 base image (includes conda). Using a specific tag for reproducibility:
FROM continuumio/miniconda3:latest

# Set working directory in container
WORKDIR /home
    
# Install Python 3.12 in a new conda environment named "wisdom"
# and install project dependencies into this environment.
COPY requirements.txt ./requirements.txt
RUN conda create -y -n wisdom python=3.12 \
     && conda run -n wisdom pip install -r requirements.txt
    
# Activate the "wisdom" environment by adding it to PATH for subsequent commands and container start
ENV PATH /opt/conda/envs/wisdom/bin:$PATH
    
# Copy the entire project code into the container (after setting up env to leverage caching)
COPY . .
    
# Set the default command (can be overridden) – for example, open a bash shell
CMD ["/bin/bash"]
    