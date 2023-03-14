# Spark-Dask Project

This project aims at implementing :
- Iris classification and 
- Parallelisation of the image processing algorithm « MedianFilter »

in Spark and Dask environements.

This repo contains both folders.

### Spark

```
cd spark/Docker
docker-compose up
```

Connect to JupiterLab NoteBook at http://localhost:8888

Both implementations run through a jupyter notebook.

### Dask

```
cd dask
conda env create -f dask.yml
conda activate dask
```

Both implementations run through a python code.
