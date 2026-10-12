# -*- coding: utf-8 -*-
"""ELEVATE Amsterdam demonstration: back end (FastAPI, CPU only).

A small API over the precomputed data in ../data (and, in development, the static pages):
  GET  /api/health
  GET  /api/search?q=...&k=12[&lat=&lon=&radius_m=]   text search over 99,100 street views
  GET  /api/near?lat=&lon=&k=8&max_m=150               views nearest to a point
  GET  /api/view/{image_id}                            one view with its observations
  GET  /api/qa                                         precomputed grounded answers
  GET  /api/ask?q=...                                  a precomputed answer, or the matching views
  POST /api/similar                                    views most similar to an uploaded photo
  /api/survey/...                                      survey answers in, totals out (app/survey.py)
Search embeds the query with EmbeddingGemma 2 on the CPU, exactly as the index was built
(GeoAI-VLM: query prompt, 256-d Matryoshka, L2-normalised). Observations are model output,
not measurements. Street images stay on the City of Amsterdam's servers; records link to them.
"""
from __future__ import annotations

import io
import json
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import os

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODEL = "google/embeddinggemma-2"
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
DIM = 256
MIN_SIMILARITY = 0.6
MAX_UPLOAD = 8 * 1024 * 1024

# The published pages live on the author's website and call this API from there.
SITE_URL = os.environ.get("ELEVATE_SITE_URL", "https://www.yunusserhat.com/elevate/")
ORIGINS = os.environ.get("ELEVATE_ORIGINS", "https://www.yunusserhat.com,https://yunusserhat.com,"
                         "http://localhost:1313,http://127.0.0.1:1313,http://localhost:7861").split(",")

app = FastAPI(title="ELEVATE Amsterdam demonstration", docs_url="/api/docs", redoc_url=None)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in ORIGINS if o.strip()],
                   allow_methods=["GET", "POST"], allow_headers=["*"], max_age=3600)


@app.middleware("http")
async def revalidate_site(request, call_next):
    """Pages, styles and scripts change between versions: browsers check them each time (the ETag keeps that cheap)."""
    response = await call_next(request)
    if not request.url.path.startswith(("/api", "/data")):
        response.headers.setdefault("Cache-Control", "no-cache")
    return response

REC = pd.read_parquet(DATA / "index/records.parquet")
REC["timestamp"] = REC["timestamp"].astype(str)
VEC = np.load(DATA / "index/vectors_f16.npy").astype(np.float32)
VEC /= np.linalg.norm(VEC, axis=1, keepdims=True)
LAT, LON = REC.lat.to_numpy(float), REC.lon.to_numpy(float)
POS = {k: i for i, k in enumerate(REC.image_id)}
QA_PATH = DATA / "tables/wp6_qa_precomputed.json"
QA = json.loads(QA_PATH.read_text()) if QA_PATH.exists() else {"meta": {}, "items": []}
FIELDS = ["image_id", "pano_id", "view", "lat", "lon", "heading_deg", "timestamp", "buurtnaam", "usable",
          "observation_text", "city_image"]

_model = None
_lock = threading.Lock()


def model():
    """EmbeddingGemma 2 on the CPU in float32, loaded on first use."""
    global _model
    with _lock:
        if _model is None:
            import torch
            from sentence_transformers import SentenceTransformer
            torch.set_num_threads(2)
            _model = SentenceTransformer(MODEL, revision=REVISION, device="cpu", model_kwargs={"dtype": torch.float32},
                                         config_kwargs={"audio_config": None})
    return _model


def embed_query(text: str) -> np.ndarray:
    v = model().encode([text], prompt_name="query", normalize_embeddings=True, truncate_dim=DIM, convert_to_numpy=True)[0]
    return v / np.linalg.norm(v)


def embed_image(img) -> np.ndarray:
    v = model().encode([{"image": img}], normalize_embeddings=True, truncate_dim=DIM, convert_to_numpy=True)[0]
    return v / np.linalg.norm(v)


def haversine_m(lat, lon, lats, lons):
    p1, p2 = np.radians(lat), np.radians(lats)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lons - lon) / 2) ** 2
    return 2 * 6371000.0 * np.arcsin(np.sqrt(a))


def rows(idx, extra: dict | None = None) -> list[dict]:
    out = REC.iloc[idx][FIELDS].copy()
    for k, v in (extra or {}).items():
        out[k] = np.round(v, 4)
    return json.loads(out.to_json(orient="records"))


def rank(v: np.ndarray, k: int, mask: np.ndarray | None = None):
    sims = VEC @ v
    if mask is not None:
        sims = np.where(mask, sims, -np.inf)
    order = np.argsort(-sims, kind="stable")[:k]
    order = order[np.isfinite(sims[order])]
    return order, sims[order]


@app.get("/api/health")
def health():
    return {"views": int(len(REC)), "model_loaded": _model is not None, "qa_items": len(QA["items"])}


@app.get("/api/search")
def search(q: str = Query(..., min_length=2, max_length=300), k: int = Query(12, ge=1, le=50),
           lat: float | None = None, lon: float | None = None, radius_m: float = Query(500, gt=0, le=5000)):
    mask = None
    if lat is not None and lon is not None:
        mask = haversine_m(lat, lon, LAT, LON) <= radius_m
        if not mask.any():
            return {"query": q, "results": [], "note": "no indexed view within the radius"}
    order, sims = rank(embed_query(q), k, mask)
    return {"query": q, "results": rows(order, {"similarity": sims}),
            "note": "Ranked by similarity of the query to each view's image and observation text. Observations are model output."}


@app.get("/api/near")
def near(lat: float = Query(..., ge=52.2, le=52.5), lon: float = Query(..., ge=4.7, le=5.1),
         k: int = Query(8, ge=1, le=40), max_m: float = Query(150, gt=0, le=2000)):
    d = haversine_m(lat, lon, LAT, LON)
    order = np.argsort(d, kind="stable")[:k]
    order = order[d[order] <= max_m]
    return {"results": rows(order, {"distance_m": d[order]})}


@app.get("/api/view/{image_id}")
def view(image_id: str):
    i = POS.get(image_id)
    if i is None:
        raise HTTPException(404, "unknown view")
    return rows([i])[0]


@app.get("/api/qa")
def qa():
    return QA


@app.get("/api/ask")
def ask(q: str = Query(..., min_length=2, max_length=300)):
    for item in QA["items"]:
        if item["question"].strip().lower() == q.strip().lower():
            return {"mode": "precomputed", **item}
    order, sims = rank(embed_query(q), 8)
    keep = sims >= MIN_SIMILARITY
    if not keep.any():
        return {"mode": "extractive", "question": q, "answer": "", "refused": True,
                "reason": f"no view reaches similarity {MIN_SIMILARITY}", "records": []}
    return {"mode": "extractive", "question": q, "answer": "", "refused": False,
            "reason": "no generation on this server: the matching views and their observations are listed",
            "records": rows(order[keep], {"similarity": sims[keep]})}


@app.post("/api/similar")
async def similar(file: UploadFile = File(...), k: int = Query(8, ge=1, le=30)):
    from PIL import Image, UnidentifiedImageError
    data = await file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "image larger than 8 MB")
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(400, "not an image")
    img.thumbnail((1024, 1024))
    order, sims = rank(embed_image(img), k)
    return {"results": rows(order, {"similarity": sims}),
            "note": "Most similar Amsterdam views by image embedding. The uploaded photo is not audited and not stored."}


# load the embedding model in the background at start, so the first search does not wait for it
threading.Thread(target=model, daemon=True).start()

# survey layer (pilot): answers reach the database only through this server
from app.survey import router as survey_router  # noqa: E402

app.include_router(survey_router)

if (ROOT / "static").exists():
    # development: the same server also serves the pages and their data
    app.mount("/data", StaticFiles(directory=DATA), name="data")
    # pages and assets last, so the API routes above keep priority
    app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="site")
else:
    # the Space serves only the API; visitors who open it are sent to the site
    @app.get("/", include_in_schema=False)
    def to_site():
        return RedirectResponse(SITE_URL)
