"""
main.py — FastAPI backend for Gradient Descent & Hooke's Law Web App
Run: uvicorn main:app --reload --port 8000
"""

import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel, Field

os.makedirs("output", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app = FastAPI(
    title="Gradient Descent & Hooke's Law Demo",
    description="Visualizes Gradient Descent algorithm and applies it to Hooke's Law via TensorFlow.",
    version="1.0.0",
)

app.mount("/output", StaticFiles(directory="output"), name="output")


# ── HTML page ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = Path("templates/index.html")
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


# ── Gradient Descent API ──────────────────────────────────────────────────────

class GDRequest(BaseModel):
    start_x: float = Field(default=-4.0, ge=-10.0, le=10.0)
    learning_rate: float = Field(default=0.1, gt=0, le=2.0)
    n_steps: int = Field(default=20, ge=5, le=100)


@app.post("/api/gd/run")
async def run_gd(req: GDRequest):
    """Run gradient descent on y=x² and generate visualization PNG."""
    try:
        from gd_vis import run_gradient_descent, plot_gd_path, plot_learning_rate_comparison
        result = run_gradient_descent(
            start_x=req.start_x,
            learning_rate=req.learning_rate,
            n_steps=req.n_steps,
        )
        gd_plot = plot_gd_path(result)
        lr_plot = plot_learning_rate_comparison()
        return JSONResponse(content={
            "success": True,
            **result,
            "plots": {
                "gd_path": "/output/gd_path.png",
                "lr_comparison": "/output/lr_comparison.png",
            },
        })
    except Exception as e:
        return JSONResponse(
            content={"success": False, "error": str(e)},
            status_code=500,
        )


# ── Hooke's Law / TensorFlow API ──────────────────────────────────────────────

class TrainRequest(BaseModel):
    epochs: int = Field(default=500, ge=50, le=2000)
    learning_rate: float = Field(default=0.01, gt=0, le=1.0)
    batch_size: int = Field(default=32, ge=8, le=256)
    n_samples: int = Field(default=500, ge=100, le=2000)
    noise_std: float = Field(default=0.05, ge=0.0, le=1.0)


@app.post("/api/hooke/train")
async def train_hooke(req: TrainRequest):
    """Train TensorFlow model for Hooke's Law."""
    try:
        from hooke_model import train_model
        result = train_model(
            epochs=req.epochs,
            learning_rate=req.learning_rate,
            batch_size=req.batch_size,
            n_samples=req.n_samples,
            noise_std=req.noise_std,
        )
        # Convert plot paths to URL paths
        url_plots = []
        for p in result.get("plots", []):
            filename = os.path.basename(p)
            url_plots.append(f"/output/{filename}")
        result["plots"] = url_plots
        return JSONResponse(content={"success": True, **result})
    except Exception as e:
        return JSONResponse(
            content={"success": False, "error": str(e)},
            status_code=500,
        )


class PredictRequest(BaseModel):
    mass_kg: float = Field(gt=0, le=100.0)


@app.post("/api/hooke/predict")
async def predict_hooke(req: PredictRequest):
    """Predict spring displacement for a given mass."""
    try:
        from hooke_model import predict
        result = predict(req.mass_kg)
        result["plot"] = f"/output/{os.path.basename(result['plot'])}"
        return JSONResponse(content={"success": True, **result})
    except RuntimeError as e:
        return JSONResponse(
            content={"success": False, "error": str(e)},
            status_code=400,
        )
    except Exception as e:
        return JSONResponse(
            content={"success": False, "error": str(e)},
            status_code=500,
        )


@app.get("/api/hooke/status")
async def model_status():
    """Check if model is trained."""
    from hooke_model import _model, _train_result
    return JSONResponse(content={
        "trained": _model is not None,
        "result": _train_result,
    })


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8003, reload=True)
