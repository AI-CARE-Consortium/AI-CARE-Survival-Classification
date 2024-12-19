# Use the official miniconda3 image as a base
FROM docker.io/continuumio/miniconda3

# Set the working directory
WORKDIR /app
# Copy the environment.yml file to the working directory
COPY environment.yml .

# Create the conda environment from the environment.yml file
RUN conda env create -f environment.yml

RUN conda init





# Copy the run.sh script to the working directory
COPY run.sh .

COPY main.py .

COPY data_import/ ./data_import/

RUN mkdir catboost_info
RUN chmod +rw catboost_info

# Make the run.sh script executable
RUN chmod +x run.sh

ENV data_path="/app/data/"

ENV entity="lung"
# Run the run.sh script
#CMD ["conda activate aicare_binary_classification"]
# Make RUN commands use the new environment
ENTRYPOINT ["conda", "run", "--live-stream", "-n", "aicare_binary_classification", "/bin/bash", "-c", "'./run.sh'"]