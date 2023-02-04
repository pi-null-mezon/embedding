# -----------------------------------------------------------------
# (C) 2023 Alex A. Taranov, Moscow, Russia, taransanya@pi-mezon.ru
# -----------------------------------------------------------------
import base64
import os
import numpy
import cv2
import torch
import aiphoto
import io

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn

app = FastAPI(
    title="Embedding API",
    description="HTTP service to extract and compare embeddings",
    version="1.0.0",
    openapi_tags=[{"name": "API"}]
)


class HealthyModel(BaseModel):
    healthy: bool = Field(description="result of healthcheck")


class UnhealthyModel(BaseModel):
    healthy: bool = Field(description="result of healthcheck", default=False)


class AboutModel(BaseModel):
    description: str = Field(description="service description")
    version: str = Field(description="version")


class ErrorModel(BaseModel):
    error: str = Field(description="error description")


class EmbeddingModel(BaseModel):
    embedding: str = Field(description="embedding in base64")
    id: str = Field(description="embedder identifier")


class CompareModel(BaseModel):
    similarity: float = Field(description="cosine similarity [-1.0 - completely different, 1.0 - very similar]")


embedders = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@app.on_event('startup')
async def startup_event():
    print(f" - hardware backend: '{device}'", flush=True)
    global embedders
    algos = os.getenv('EMBEDDERS', 'TorchvisionResNet50')
    if 'TorchvisionResNet50' in algos:
        embedders.append(aiphoto.TorchvisionResNet50(device))
    if 'TorchvisionEfficientNetB5' in algos:
        embedders.append(aiphoto.TorchvisionEfficientNetB5(device))
    if 'TorchvisionMobileNetV2' in algos:
        embedders.append(aiphoto.TorchvisionMobileNetV2(device))


@app.get(f"/about",
         response_class=JSONResponse,
         tags=["API"],
         response_model=AboutModel,
         summary="ask short description")
async def about():
    return {"description": app.description, "version": app.version}


@app.get(f"/healthcheck",
         response_class=JSONResponse,
         tags=["API"],
         response_model=HealthyModel,
         responses={500: {"model": UnhealthyModel}},
         summary="check service health")
async def get_status():
    uid, orange = aiphoto.embedding(embedders, cv2.imread("./samples/orange.jpg", cv2.IMREAD_COLOR))
    uid, starship = aiphoto.embedding(embedders, cv2.imread("./samples/starship.jpg", cv2.IMREAD_COLOR))
    similarity = float(aiphoto.similarity(orange, starship))
    print(f"healthcheck similarity: {similarity:.3f}")
    return {"healthy": bool(similarity < 0.5)}


@app.post("/extract",
          response_class=JSONResponse,
          response_description="embedding",
          tags=["API"],
          responses={400: {"model": ErrorModel}, 500: {"model": ErrorModel}},
          summary="extract embedding")
async def extract(file: UploadFile = File(None, description="file to extract embedding")):
    target_modality = 'photo' if 'image' in file.content_type else None
    if target_modality is None:
        return JSONResponse({"error": f"Do not know what to do with {file.content_type }"}, status_code=400)
    bin_data = await file.read()
    photo = cv2.imdecode(numpy.frombuffer(bin_data, dtype=numpy.uint8), cv2.IMREAD_COLOR)
    if not isinstance(photo, numpy.ndarray):
        return JSONResponse({"error": "Can not decode photo"}, status_code=400)
    uid, emb = aiphoto.embedding(embedders, photo)
    return {"embedding": serialize(emb), "id": uid}


@app.post("/compare",
          response_class=JSONResponse,
          response_description="embedding",
          tags=["API"],
          responses={400: {"model": ErrorModel}},
          summary="extract embedding")
async def compare(first: EmbeddingModel, second: EmbeddingModel ):
    if first.id != second.id:
        return JSONResponse({"error": "Embeddings identifiers do not match"}, status_code=400)
    similarity = float(aiphoto.similarity(deserialize(first.embedding), deserialize(second.embedding)))
    return {"similarity": similarity}


def serialize(t: torch.Tensor) -> str:
    return base64.b64encode(t.cpu().numpy().tobytes()).decode('utf-8')


def deserialize(s: str) -> torch.Tensor:
    a = numpy.copy(numpy.frombuffer(base64.b64decode(s.encode('utf-8')), dtype=numpy.float32))
    return torch.from_numpy(a).unsqueeze(0)


if __name__ == '__main__':
    print('='*len(app.description), flush=True)
    print(app.description, flush=True)
    print('='*len(app.description), flush=True)
    print(f'Version: {app.version}', flush=True)
    print('Configuration:', flush=True)
    http_srv_addr = os.getenv("APP_ADDR", "0.0.0.0")
    http_srv_port = int(os.getenv("APP_PORT", 5000))
    workers = int(os.getenv("WORKERS", 1))
    print(f" - workers: {workers}", flush=True)
    uvicorn.run('httpsrv:app', host=http_srv_addr, port=http_srv_port, workers=workers)
