from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import json
import redis
import asyncio
from datetime import datetime
import numpy as np

app = FastAPI(
    title="DubIndex API",
    description="Global LLM Ranking System using the DubIndex Formula",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis setup for caching
redis_client = redis.Redis(host='localhost', port=6379, db=0)

class ModelScore(BaseModel):
    """Model for submitting new scores."""
    model_name: str
    speed: float
    hallucination_resistance: float
    accuracy: float
    efficiency: float
    size: float
    timestamp: Optional[datetime] = None
    submitter: Optional[str] = None

class ModelRanking(BaseModel):
    """Model for ranking response."""
    model_name: str
    dub_index: float
    components: Dict[str, float]
    last_updated: datetime

def calculate_dub_index(score: ModelScore) -> float:
    """Calculate DubIndex score using the official formula."""
    # Normalize values
    speed_norm = (score.speed - 100) / 900  # Assuming range 100-1000 tokens/sec
    hall_norm = 1 / (1 + score.hallucination_resistance)
    acc_norm = (score.accuracy - 0.8) / 0.19  # Assuming range 0.8-0.99
    eff_norm = (score.efficiency - 0.7) / 0.25  # Assuming range 0.7-0.95
    size_norm = 1 / (1 + score.size)

    # DubIndex formula
    dub_index = (
        0.20 * speed_norm +
        0.30 * hall_norm +
        0.30 * acc_norm +
        0.10 * eff_norm +
        0.10 * size_norm
    )
    
    return round(dub_index, 2)

@app.get("/")
async def root():
    """API root with documentation link."""
    return {
        "message": "Welcome to DubIndex API",
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }

@app.post("/submit_score")
async def submit_score(score: ModelScore):
    """Submit a new score for an LLM model."""
    try:
        # Calculate DubIndex
        dub_index = calculate_dub_index(score)
        
        # Store in Redis with timestamp
        data = {
            "dub_index": dub_index,
            "components": {
                "speed": score.speed,
                "hallucination_resistance": score.hallucination_resistance,
                "accuracy": score.accuracy,
                "efficiency": score.efficiency,
                "size": score.size
            },
            "timestamp": datetime.now().isoformat(),
            "submitter": score.submitter
        }
        
        redis_client.hset(
            "model_scores",
            score.model_name,
            json.dumps(data)
        )
        
        return {
            "model_name": score.model_name,
            "dub_index": dub_index,
            "status": "Score submitted successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rankings")
async def get_rankings():
    """Get current rankings of all models."""
    try:
        # Get all scores from Redis
        scores = redis_client.hgetall("model_scores")
        
        # Parse and sort rankings
        rankings = []
        for model_name, score_data in scores.items():
            data = json.loads(score_data)
            rankings.append({
                "model_name": model_name.decode(),
                "dub_index": data["dub_index"],
                "components": data["components"],
                "last_updated": data["timestamp"]
            })
        
        # Sort by DubIndex score
        rankings.sort(key=lambda x: x["dub_index"], reverse=True)
        
        return {
            "rankings": rankings,
            "total_models": len(rankings),
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/{model_name}")
async def get_model_score(model_name: str):
    """Get detailed score for a specific model."""
    try:
        score_data = redis_client.hget("model_scores", model_name)
        if not score_data:
            raise HTTPException(status_code=404, detail="Model not found")
            
        data = json.loads(score_data)
        return {
            "model_name": model_name,
            "dub_index": data["dub_index"],
            "components": data["components"],
            "last_updated": data["timestamp"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/compare")
async def compare_models(model_names: List[str]):
    """Compare multiple models."""
    try:
        comparisons = []
        for model_name in model_names:
            score_data = redis_client.hget("model_scores", model_name)
            if score_data:
                data = json.loads(score_data)
                comparisons.append({
                    "model_name": model_name,
                    "dub_index": data["dub_index"],
                    "components": data["components"]
                })
        
        return {
            "comparisons": comparisons,
            "total_compared": len(comparisons)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
