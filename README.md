embedding
===

HTTP service to extract and compare embeddings

## Build docker image:

```
docker build -t pimezon/embedding -f Dockerfile_CPU .
```

or for CUDA accelerated version:

```
docker build -t pimezon/embedding -f Dockerfile_CUDA .
```

## Run composition

```
docker-compose up
```

## Check and try out API

open http://localhost:8080/docs in browser


