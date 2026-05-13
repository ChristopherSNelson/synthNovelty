"""
FastAPI server for the SynthNovelty reaction novelty scorer.

Endpoints:
    GET  /health            Model status
    POST /score             Score one or more reactions (single timestep or multi-t)
    POST /score_route       Score a multi-step synthesis route

Install dependencies:
    pip install fastapi uvicorn

Run:
    uvicorn api:app --reload --port 8000

Example:
    curl -X POST http://localhost:8000/score \
         -H "Content-Type: application/json" \
         -d '{"reactions": ["CCBr.Nc1ccccc1>>CCNc1ccccc1"]}'
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import os


# ---- State -------------------------------------------------------------------

_scorer: dict = {}  # populated at startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists("model.pt") or not os.path.exists("metrics.pt"):
        raise RuntimeError(
            "model.pt or metrics.pt not found. Run the full training pipeline first."
        )
    from score_reaction import load_model, score_reactions, score_reactions_multit
    model, mean, std, rxnfp_gen, mean_freq, device = load_model()
    _scorer["model"] = model
    _scorer["mean"] = mean
    _scorer["std"] = std
    _scorer["rxnfp_gen"] = rxnfp_gen
    _scorer["mean_freq"] = mean_freq
    _scorer["device"] = device
    _scorer["score"] = score_reactions
    _scorer["score_multit"] = score_reactions_multit
    yield
    _scorer.clear()


app = FastAPI(
    title="SynthNovelty API",
    description="Reaction-space novelty scoring via conditional diffusion on RXNFP embeddings.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---- Schemas -----------------------------------------------------------------

class ScoreRequest(BaseModel):
    reactions: List[str] = Field(..., description="List of reaction SMILES (reactants>>product)")
    timesteps: List[float] = Field(
        default=[0.5],
        description="Diffusion timestep(s). Multiple values average scores and add uncertainty."
    )


class ScoredReaction(BaseModel):
    smiles: str
    novelty: float
    uncertainty: Optional[float] = Field(
        None, description="Std across timesteps (only present when multiple timesteps given)"
    )


class ScoreRouteRequest(BaseModel):
    steps: List[str] = Field(..., description="Ordered list of reaction SMILES (first -> last step)")


class RouteResult(BaseModel):
    step_scores: List[float]
    mean_novelty: float
    max_novelty: float
    min_novelty: float
    num_steps: int


# ---- Endpoints ---------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bool(_scorer)}


@app.post("/score", response_model=List[ScoredReaction])
def score(req: ScoreRequest):
    if not _scorer:
        raise HTTPException(503, "Model not loaded")
    if not req.reactions:
        raise HTTPException(400, "reactions list is empty")
    if not all(">>" in r for r in req.reactions):
        raise HTTPException(400, "All reactions must contain '>>'")

    m, mean, std, gen, mf, dev = (
        _scorer["model"], _scorer["mean"], _scorer["std"],
        _scorer["rxnfp_gen"], _scorer["mean_freq"], _scorer["device"],
    )

    try:
        if len(req.timesteps) > 1:
            scores, uncertainties = _scorer["score_multit"](
                req.reactions, m, mean, std, gen, mf, dev, req.timesteps)
        else:
            scores = _scorer["score"](
                req.reactions, m, mean, std, gen, mf, dev, req.timesteps[0])
            uncertainties = [None] * len(scores)
    except Exception as e:
        raise HTTPException(500, f"Scoring failed: {e}")

    return [
        ScoredReaction(
            smiles=rxn,
            novelty=float(s),
            uncertainty=float(u) if u is not None else None,
        )
        for rxn, s, u in zip(req.reactions, scores,
                              uncertainties if len(req.timesteps) > 1 else [None] * len(scores))
    ]


@app.post("/score_route", response_model=RouteResult)
def score_route(req: ScoreRouteRequest):
    if not _scorer:
        raise HTTPException(503, "Model not loaded")
    if not req.steps:
        raise HTTPException(400, "steps list is empty")
    if not all(">>" in s for s in req.steps):
        raise HTTPException(400, "All steps must be reaction SMILES containing '>>'")

    m, mean, std, gen, mf, dev = (
        _scorer["model"], _scorer["mean"], _scorer["std"],
        _scorer["rxnfp_gen"], _scorer["mean_freq"], _scorer["device"],
    )

    try:
        step_scores = _scorer["score"](req.steps, m, mean, std, gen, mf, dev)
    except Exception as e:
        raise HTTPException(500, f"Scoring failed: {e}")

    return RouteResult(
        step_scores=[float(s) for s in step_scores],
        mean_novelty=float(np.mean(step_scores)),
        max_novelty=float(np.max(step_scores)),
        min_novelty=float(np.min(step_scores)),
        num_steps=len(step_scores),
    )
