from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from project_final_version import predict_crop
import numpy as np  

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserInputs(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float
    soil_moisture: float
    organic_matter: float
    sunlight_exposure: float
    wind_speed: float
    urban_area_proximity: float
    soil_type: int
    water_source_type: int
    fertilizer_usage: float
    co2_concentration: float
    irrigation_frequency: float
    pest_pressure: float
    method: str

@app.post("/api/predict/")
def predict(data: UserInputs):
    payload = data.dict()
    method = payload.pop("method")
    try:
        print("I'm here")
        result = predict_crop(method, payload, k=3)
        # Convert numpy to native 
        def convert(o):
            if isinstance(o, np.generic):
                return o.item()
            elif isinstance(o, dict):
                return {k: convert(v) for k, v in o.items()}
            elif isinstance(o, list):
                return [convert(v) for v in o]
            else:
                return o
    
        return convert(result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
