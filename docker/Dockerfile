FROM continuumio/miniconda3

# RUN apt-get update && apt-get install -y build-essential

WORKDIR /app

COPY environment.yml .

RUN conda install -n base -c conda-forge mamba

RUN mamba env create -f environment.yml


COPY . .

ENTRYPOINT ["conda", "run", "-n", "spt", "python3"]