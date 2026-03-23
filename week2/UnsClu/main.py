"""
FastAPI backend for Hooke's Law + K-Means Unsupervised Learning Demo
Usage: uvicorn main:app --reload --port 8001
"""

import os
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np

import model as ml

# ─────────────────────────────────────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

app = FastAPI(
    title="Hooke's Law × K-Means Demo",
    description="Unsupervised clustering + TensorFlow regression for spring physics",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    epochs: int = Field(default=200, ge=50, le=500)
    learning_rate: float = Field(default=0.001)
    k_clusters: int = Field(default=3, ge=2, le=5)
    n_per_spring: int = Field(default=50, ge=20, le=200)


class PredictRequest(BaseModel):
    mass_kg: float = Field(..., ge=0.01, le=10.0)
    cluster_id: int = Field(default=0, ge=0, le=4)


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main HTML page"""
    html_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/train")
async def train(req: TrainRequest):
    """
    전체 파이프라인 실행:
    1. 데이터 생성
    2. K-Means 군집화 + PNG 저장
    3. TensorFlow 훈련 + PNG 저장
    """
    try:
        # 1. Generate data
        X, y_true, y_ext = ml.generate_spring_data(n_per_spring=req.n_per_spring)

        # 2. K-Means (unsupervised)
        cluster_labels, cluster_centers = ml.run_kmeans(X, k=req.k_clusters)

        # 3. TensorFlow training — cluster_labels 포함하여 스프링 종류 구분
        masses = X[:, 0]
        loss_hist, val_loss_hist, r2 = ml.train_tensorflow(
            masses, y_ext,
            epochs=req.epochs,
            learning_rate=req.learning_rate,
            cluster_labels=cluster_labels,
        )

        # 4. List saved plots
        plots = ml.get_plots()

        return {
            "status": "success",
            "r2_score": r2,
            "epochs_trained": len(loss_hist),
            "loss_history": loss_hist,
            "val_loss_history": val_loss_hist,
            "cluster_centers": cluster_centers.tolist(),
            "n_clusters": req.k_clusters,
            "plots": plots,
            "message": f"Training complete! R² = {r2:.4f} | "
                       f"Epochs: {len(loss_hist)} | "
                       f"K-Means: {req.k_clusters} clusters",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict")
async def predict(req: PredictRequest):
    """새 질량에 대한 스프링 늘어남 예측"""
    try:
        result = ml.predict(req.mass_kg, cluster_id=req.cluster_id)
        return {"status": "success", **result}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/plots")
async def get_plots():
    """저장된 PNG 목록 반환"""
    plots = ml.get_plots()
    return {
        "plots": plots,
        "paths": [f"/output/{p}" for p in plots],
    }


@app.get("/api/status")
async def get_status():
    """모델 훈련 상태"""
    return ml.get_status()


@app.get("/health")
async def health():
    return {"status": "ok", "service": "HookesLawKMeans"}
